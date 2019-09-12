import torch
from torch.utils import data
import numpy as np
from sklearn.metrics import r2_score, classification_report, roc_auc_score, average_precision_score

import datetime


OUTPUT_DIR = 'output/'
TENSORBOARDX_OUTPUT_DIR = 'tbxoutput/'
SAVEDMODELS_DIR = 'savedmodels/'
# time of importing this file, including microseconds because slurm may start queued jobs very close in time
DATETIME_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


class Globals: # container for all objects getting passed between log calls
    evaluate_called = False

g = Globals()

TRAIN_SUBSET_SIZE = 500
SUBSET_LOADER_BATCH_SIZE = 50

def subset_loader(dataloader, subset_size, seed=0):
    np.random.seed(seed)
    random_indices = np.random.choice(len(dataloader.dataset), subset_size)
    np.random.seed()  # "reset" seed
    subset = data.Subset(dataloader.dataset, random_indices)
    return data.DataLoader(subset, batch_size=SUBSET_LOADER_BATCH_SIZE, collate_fn=dataloader.collate_fn)


def compute_roc_auc(output, target):
    def roc_auc_of_column(scores_column, targets_column):
        relevant_indices = targets_column.nonzero()
        relevant_targets = targets_column[relevant_indices]
        relevant_scores = scores_column[relevant_indices]
        relevant_targets_np = relevant_targets.cpu().numpy()
        relevant_targets_np = relevant_targets_np == 1  # -1s/1s => Falses/Trues
        try:
            score = roc_auc_score(relevant_targets_np, relevant_scores.cpu().detach().numpy())
        except:
            score = np.nan
        return score

    scores = torch.sigmoid(output)
    roc_aucs = [
        roc_auc_of_column(scores[:, i], target[:, i])
        for i in range(target.shape[1])
    ]
    return roc_aucs

def compute_pr_auc(output, target):
    def pr_auc_of_column(scores_column, targets_column):
        relevant_indices = targets_column.nonzero()
        relevant_targets = targets_column[relevant_indices]
        relevant_scores = scores_column[relevant_indices]
        relevant_targets_np = relevant_targets.cpu().numpy()
        relevant_targets_np = relevant_targets_np == 1  # -1s/1s => Falses/Trues
        return average_precision_score(relevant_targets_np, relevant_scores.cpu().detach().numpy())

    scores = torch.sigmoid(output)
    pr_aucs = [
        pr_auc_of_column(scores[:, i], target[:, i])
        for i in range(target.shape[1])
    ]
    return pr_aucs

def compute_mse(output, target):
    nn_mse = torch.nn.MSELoss()
    mses = [
        nn_mse(output[:, i], target[:, i]).cpu().detach().numpy()
        for i in range(target.shape[1])
    ]
    return mses

def compute_rmse(output, target):
    mses = compute_mse(output, target)
    return np.sqrt(mses)

SCORE_FUNCTIONS = {
    'roc-auc': compute_roc_auc, 'pr-auc': compute_pr_auc, 'MSE': compute_mse, 'RMSE': compute_rmse
}


def feed_net(net, dataloader, criterion, cuda):
    batch_outputs = []
    batch_losses = []
    batch_targets = []
    for i_batch, batch in enumerate(dataloader):
        if cuda:
            batch = [tensor.cuda(non_blocking=True) for tensor in batch]
        adjacency, nodes, edges, target = batch
        output = net(adjacency, nodes, edges)
        loss = criterion(output, target)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(target)
    outputs = torch.cat(batch_outputs)
    loss = np.mean(batch_losses)
    targets = torch.cat(batch_targets)
    return outputs, loss, targets

def evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args):
    global g
    if not g.evaluate_called:
        g.evaluate_called = True
        if args.score == 'roc-auc' or args.score == 'pr-auc':
            g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score = 0, 0, 0
        elif args.score == 'MSE' or args.score == 'RMSE':
            # just something large, this is arbitrary
            g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score = 10, 10, 10
        #g.train_subset_loader = subset_loader(train_dataloader, TRAIN_SUBSET_SIZE, seed=0)
        g.train_subset_loader = train_dataloader

    train_output, train_loss, train_target = feed_net(net, g.train_subset_loader, criterion, args.cuda)
    validation_output, validation_loss, validation_target = feed_net(net, validation_dataloader, criterion, args.cuda)
    test_output, test_loss, test_target = feed_net(net, test_dataloader, criterion, args.cuda)

    train_scores = SCORE_FUNCTIONS[args.score](train_output, train_target)
    train_mean_score = np.nanmean(train_scores)
    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    validation_mean_score = np.nanmean(validation_scores)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)
    test_mean_score = np.nanmean(test_scores)

    if args.score == 'roc-auc' or args.score == 'pr-auc':
        new_best_model_found = validation_mean_score > g.best_mean_validation_score
    elif args.score == 'MSE' or args.score == 'RMSE':
        new_best_model_found = validation_mean_score < g.best_mean_validation_score

    if new_best_model_found:
        g.best_mean_train_score = train_mean_score
        g.best_mean_validation_score = validation_mean_score
        g.best_mean_test_score = test_mean_score

        if args.savemodel:
            path = SAVEDMODELS_DIR + type(net).__name__ + DATETIME_STR
            torch.save(net, path)

    target_names = train_dataloader.dataset.target_names
    return {  # if made deeper, tensorboardx writing breaks I think
        'loss': {'train': train_loss, 'test': test_loss},
        'mean {}'.format(args.score):
            {'train': train_mean_score, 'validation': validation_mean_score, 'test': test_mean_score},
        'train {}s'.format(args.score): {target_names[i]: train_scores[i] for i in range(len(target_names))},
        'test {}s'.format(args.score): {target_names[i]: test_scores[i] for i in range(len(target_names))},
        'best mean {}'.format(args.score):
            {'train': g.best_mean_train_score, 'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
    }


def get_run_info(net, args):
    return {
        'net': type(net).__name__,
        'args': ', '.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),
        'modules': {name: str(module) for name, module in  net._modules.items()}
    }


def less_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):
    scalars = evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args)
    mean_score_key = 'mean {}'.format(args.score)
    print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}: {}'.format(
        epoch + 1,
        args.score, scalars[mean_score_key]['train'],
        args.score, scalars[mean_score_key]['validation'],
        args.score, scalars[mean_score_key]['test'])
    )

def more_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):
    mean_score_key = 'mean {}'.format(args.score)
    best_mean_score_key = 'best {}'.format(mean_score_key)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')
        print('score metric: {}'.format(args.score))
        print('columns:')
        print(
            'epochs, ' + \
            'mean training score, mean validation score, mean testing score, ' + \
            'best-model-so-far mean training score, best-model-so-far mean validation score, best-model-so-far mean testing score'
        )

    scalars = evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args)
    print(
        '%d, %f, %f, %f, %f, %f, %f' % (
            epoch + 1,
            scalars[mean_score_key]['train'], scalars[mean_score_key]['validation'], scalars[mean_score_key]['test'],
            scalars[best_mean_score_key]['train'], scalars[best_mean_score_key]['validation'], scalars[best_mean_score_key]['test']
        )
    )

# to open tensorboard training summaries, live or static:
# 1) do some training to generate them in tbxoutput/
# 2) install tensorflow (in a separate environment is fine)
# 3) run tensorboard --port 6011 --logdir tbxoutput/ and open localhost:6011 in a browser
def tensorboardx_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):
    global g
    if not g.evaluate_called:
        from tensorboardX import SummaryWriter

        run_info = get_run_info(net, args)

        class_str = run_info['net']
        output_subdir = TENSORBOARDX_OUTPUT_DIR + class_str + ' ' + DATETIME_STR
        g.writer = SummaryWriter(output_subdir)

        g.writer.add_text('args', run_info['args'])
        for k, v in run_info['modules'].items():
            g.writer.add_text(k, v)
    else:
        #writer = SummaryWriter(output_subdir) # tensorboardx bug causes this to crash on epoch 40 or so
        g.writer.file_writer.reopen() # workaround

    scalars = evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args)

    for k, v in scalars.items():
        g.writer.add_scalars(k, v, epoch)

    #writer.close() # tensorboardx bug causes this to crash on epoch 40 or so
    g.writer.file_writer.close() # workaround

    print('epoch %d, training loss: %f, validation loss: %f' %
          (epoch + 1, scalars['loss']['train'], scalars['loss']['validation']))

LOG_FUNCTIONS = {
    'less': less_log, 'more': more_log, 'tensorboardx': tensorboardx_log
}

