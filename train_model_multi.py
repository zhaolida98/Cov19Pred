import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
import utils, validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import pandas as pd


# from scipy import interp

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def plot_training_history(loss, val_loss, acc, val_acc, fscore, val_fscore, cell_type):
    """
    Plots the loss and accuracy for training and validation over epochs.
    Also plots the logits for a small batch over epochs.
    """
    plt.style.use('ggplot')

    # Plot losses
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(loss, 'b', label='Training')
    plt.plot(val_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(acc, 'b', label='Training')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 3)
    plt.plot(fscore, 'b', label='Training')
    plt.plot(val_fscore, 'r', label='Validation')
    plt.title('F-Score')
    plt.legend()

    # plt.show()
    plt.savefig(f'./data/figure/loss_fig_{cell_type}.png', dpi=350)
    print(f'plot save to ./data/figure/loss_fig_{cell_type}.png')


def plot_attention(weights):
    """
    Plots attention weights in a grid.
    """
    cax = plt.matshow(weights.numpy(), cmap='bone')
    plt.colorbar(cax)
    plt.grid(
        b=False,
        axis='both',
        which='both',
    )
    plt.xlabel('Years')
    plt.ylabel('Examples')
    # plt.savefig('./reports/figures/attention_weights.png')
    plt.savefig('./data/figure/attention/weight.eps', dpi=350)
    print('save to ./data/figure/attention/weight.eps')


def predictions_from_output(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions


def calculate_prob(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    pred_probe, _ = prob.topk(1)
    return pred_probe


def verify_model(model, X, Y, batch_size):
    """
    Checks the loss at initialization of the model and asserts that the
    training examples in a batch aren't mixed together by backpropagating.
    """
    print('Sanity checks:')
    criterion = torch.nn.CrossEntropyLoss()
    scores, _ = model(X, model.init_hidden(Y.shape[0]))
    print(' Loss @ init %.3f, expected ~%.3f' % (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))

    mini_batch_X = X[:, :batch_size, :]
    mini_batch_X.requires_grad_()
    criterion = torch.nn.MSELoss()
    scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

    non_zero_idx = 1
    perfect_scores = [[0, 0] for i in range(batch_size)]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]

    scores.data = torch.FloatTensor(not_perfect_scores)
    Y_perfect = torch.FloatTensor(perfect_scores)
    loss = criterion(scores, Y_perfect)
    loss.backward()

    zero_tensor = torch.FloatTensor([0] * X.shape[2])
    for i in range(mini_batch_X.shape[0]):
        for j in range(mini_batch_X.shape[1]):
            if sum(mini_batch_X.grad[i, j] != zero_tensor):
                assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

    mini_batch_X.detach()
    print(' Backpropagated dependencies OK')


# def train_rnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention, cell_type):
#     """
#     Training loop for a model utilizing hidden states.

#     verify enables sanity checks of the model.
#     epochs decides the number of training iterations.
#     learning rate decides how much the weights are updated each iteration.
#     batch_size decides how many examples are in each mini batch.
#     show_attention decides if attention weights are plotted.
#     """
#     print_interval = 10
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     criterion = torch.nn.CrossEntropyLoss()
#     num_of_examples = X.shape[1]
#     num_of_batches = math.floor(num_of_examples / batch_size)

#     if verify:
#         verify_model(model, X, Y, batch_size)
#     all_losses = []
#     all_val_losses = []
#     all_accs = []
#     all_val_accs = []
#     all_pres = []
#     all_recs = []
#     all_fscores = []
#     all_val_fscores = []
#     all_mccs = []

#     best_val_loss = 100000000.0
#     best_val_acc = 0.0
#     best_val_pre = 0.0
#     best_val_rec = 0.0
#     best_val_fscore = 0.0
#     best_val_mcc = 0.0
#     best_epoch_index = 0

#     # Find mini batch that contains at least one mutation to plot
#     plot_batch_size = 10
#     i = 0
#     while not Y_test[i]:
#         i += 1

#     X_plot_batch = X_test[:, i:i + plot_batch_size, :]
#     Y_plot_batch = Y_test[i:i + plot_batch_size]
#     plot_batch_scores = []

#     start_time = time.time()
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0
#         running_acc = 0
#         running_pre = 0
#         running_pre_total = 0
#         running_rec = 0
#         running_rec_total = 0
#         epoch_fscore = 0
#         running_mcc_numerator = 0
#         running_mcc_denominator = 0
#         running_rec_total = 0

#         hidden = model.init_hidden(batch_size)

#         for count in range(0, num_of_examples - batch_size + 1, batch_size):
#             repackage_hidden(hidden)

#             X_batch = X[:, count:count + batch_size, :]
#             Y_batch = Y[count:count + batch_size]

#             scores, _ = model(X_batch, hidden)

#             # scores = model(X_batch)

#             loss = criterion(scores, Y_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             predictions = predictions_from_output(scores)

#             conf_matrix = validation.get_confusion_matrix(Y_batch, predictions)
#             conf_matrix = np.array(conf_matrix)
#             total_samples = np.sum(conf_matrix)
#             # 计算整体的 TP、FP、FN 和 TN （macro）
#             TP = np.trace(conf_matrix)  # 对角线元素之和
#             FP = np.sum(conf_matrix.sum(axis=0) - np.diag(conf_matrix))  # 每列总和减去对角线元素，然后求和
#             FN = np.sum(conf_matrix.sum(axis=1) - np.diag(conf_matrix))  # 每行总和减去对角线元素，然后求和
#             TN = total_samples - (TP + FP + FN)  # 总样本数减去 TP, FP 和 FN
#             running_acc += TP + TN
#             running_pre += TP
#             running_pre_total += TP + FP
#             running_rec += TP
#             running_rec_total += TP + FN
#             running_mcc_numerator += (TP * TN - FP * FN)
#             if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
#                 running_mcc_denominator += 0
#             else:
#                 running_mcc_denominator += math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
#             running_loss += loss.item()

#         elapsed_time = time.time() - start_time
#         epoch_acc = running_acc / Y.shape[0]
#         all_accs.append(epoch_acc)

#         if running_pre_total == 0:
#             epoch_pre = 0
#         else:
#             epoch_pre = running_pre / running_pre_total
#         all_pres.append(epoch_pre)

#         if running_rec_total == 0:
#             epoch_rec = 0
#         else:
#             epoch_rec = running_rec / running_rec_total
#         all_recs.append(epoch_rec)

#         if (epoch_pre + epoch_rec) == 0:
#             epoch_fscore = 0
#         else:
#             epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
#         all_fscores.append(epoch_fscore)

#         if running_mcc_denominator == 0:
#             epoch_mcc = 0
#         else:
#             epoch_mcc = running_mcc_numerator / running_mcc_denominator
#         all_mccs.append(epoch_mcc)

#         epoch_loss = running_loss / num_of_batches
#         all_losses.append(epoch_loss)

#         with torch.no_grad():
#             model.eval()
#             test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))

#             #####
#             # prob_out = F.softmax(test_scores,dim=1)
#             # # 将张量转换为 NumPy 数组
#             # prob_np = prob_out.detach().cpu().numpy()

#             # # 转换为 Pandas DataFrame
#             # prob_df = pd.DataFrame(prob_np)
#             # prob_df.to_csv("/scratch/project16/TEMPO-ESM/mycode/RNN_prob_out_3gram.txt", sep="\t", index=False)
#             #####

#             # test_scores = model(X_test)

#             predictions = predictions_from_output(test_scores)
#             predictions = predictions.view_as(Y_test)
#             pred_prob = calculate_prob(test_scores)
#             # precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, predictions)
#             # pd.DataFrame(Y_test.detach().cpu().numpy()).to_csv("/scratch/project16/TEMPO-ESM/mycode/RNN_ytest_3gram.txt", sep="\t", index=False)
#             # pd.DataFrame(predictions.detach().cpu().numpy()).to_csv("/scratch/project16/TEMPO-ESM/mycode/RNN_predictions_3gram.txt", sep="\t", index=False)

#             # Convert tensors to numpy for evaluation
#             predictions = predictions.cpu().numpy()
#             y_true = Y_test.cpu().numpy()

#             # Validation metrics
#             val_loss = criterion(test_scores, Y_test).item()
#             val_acc = accuracy_score(y_true, predictions)
#             precision = precision_score(y_true, predictions, average='micro')
#             recall = recall_score(y_true, predictions, average='micro')
#             fscore = f1_score(y_true, predictions, average='micro')
#             mcc = matthews_corrcoef(y_true, predictions)

#             all_val_losses.append(val_loss)
#             all_val_accs.append(val_acc)
#             all_val_fscores.append(fscore)

#             if val_acc>best_val_acc:
#                 best_val_loss = val_loss
#                 best_val_acc = val_acc
#                 best_val_pre = precision
#                 best_val_rec = recall
#                 best_val_fscore = fscore
#                 best_val_mcc = mcc
#                 best_epoch_index = epoch

#             # plot_scores, _ = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
#             # plot_batch_scores.append(plot_scores)

#         if (epoch + 1) % print_interval == 0:
#             print('Epoch %d Time %s' % (epoch, utils.get_time_string(elapsed_time)))
#             print('T_loss %.3f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' % (
#             epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
#             print('V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
#             val_loss, val_acc, precision, recall, fscore, mcc))
#     plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, all_fscores, all_val_fscores, cell_type)
#     print('Best results: %d \n V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
#         best_epoch_index, best_val_loss, best_val_acc, best_val_pre, best_val_rec, best_val_fscore, best_val_mcc))
#     # roc curve
#     # if epoch + 1 == 50:
#     #   tpr_rnn, fpr_rnn, _ = roc_curve(Y_test, pred_prob)
#     #   print(auc(fpr_rnn, tpr_rnn))
#     #   plt.figure(1)
#     #   #plt.xlim(0, 0.8)
#     #   plt.ylim(0.5, 1)
#     #   plt.plot([0, 1], [0, 1], 'k--')
#     #   if cell_type == 'lstm':
#     #       plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
#     #   elif cell_type == 'rnn':
#     #       plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
#     #   elif cell_type == 'gru':
#     #       plt.plot(fpr_rnn, tpr_rnn, label='attention')
#     #   elif cell_type == 'attention':
#     #       plt.plot(fpr_rnn, tpr_rnn, label='gru')
#     #   plt.legend(loc='best')

#     # if show_attention:
#     #   with torch.no_grad():
#     #     model.eval()
#     #     _, attn_weights = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
#     #     plot_attention(attn_weights)
#     # plt.show()

def train_rnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention, cell_type):
    """
    Training loop for a model utilizing hidden states.

    verify enables sanity checks of the model.
    epochs decides the number of training iterations.
    learning rate decides how much the weights are updated each iteration.
    batch_size decides how many examples are in each mini batch.
    show_attention decides if attention weights are plotted.
    """
    print_interval = epochs // 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    num_of_examples = X.shape[1]
    num_of_batches = math.floor(num_of_examples / batch_size)

    if verify:
        verify_model(model, X, Y, batch_size)
    all_losses = []
    all_val_losses = []
    all_accs = []
    all_val_accs = []
    all_pres = []
    all_recs = []
    all_fscores = []
    all_val_fscores = []
    all_mccs = []

    best_val_loss = 100000000.0
    best_val_acc = 0.0
    best_val_pre = 0.0
    best_val_rec = 0.0
    best_val_fscore = 0.0
    best_val_mcc = 0.0
    best_epoch_index = 0

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        all_train_predictions = []
        all_train_true = []

        hidden = model.init_hidden(batch_size)

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            repackage_hidden(hidden)

            X_batch = X[:, count:count + batch_size, :]
            Y_batch = Y[count:count + batch_size]

            scores, _ = model(X_batch, hidden)

            loss = criterion(scores, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = predictions_from_output(scores)
            all_train_predictions.extend(predictions.cpu().numpy())
            all_train_true.extend(Y_batch.cpu().numpy())

            running_loss += loss.item()

        # Calculate metrics on the entire training dataset for this epoch
        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        epoch_acc = accuracy_score(all_train_true, all_train_predictions)
        all_accs.append(epoch_acc)

        epoch_pre = precision_score(all_train_true, all_train_predictions, average='micro')
        all_pres.append(epoch_pre)

        epoch_rec = recall_score(all_train_true, all_train_predictions, average='micro')
        all_recs.append(epoch_rec)

        epoch_fscore = f1_score(all_train_true, all_train_predictions, average='micro')
        all_fscores.append(epoch_fscore)

        epoch_mcc = matthews_corrcoef(all_train_true, all_train_predictions)
        all_mccs.append(epoch_mcc)

        # Validation metrics
        with torch.no_grad():
            model.eval()
            test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))

            predictions = predictions_from_output(test_scores)
            predictions = predictions.view_as(Y_test)

            y_true = Y_test.cpu().numpy()
            predictions = predictions.cpu().numpy()

            val_loss = criterion(test_scores, Y_test).item()
            val_acc = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='micro')
            recall = recall_score(y_true, predictions, average='micro')
            fscore = f1_score(y_true, predictions, average='micro')
            mcc = matthews_corrcoef(y_true, predictions)

            all_val_losses.append(val_loss)
            all_val_accs.append(val_acc)
            all_val_fscores.append(fscore)

            if val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_pre = precision
                best_val_rec = recall
                best_val_fscore = fscore
                best_val_mcc = mcc
                best_epoch_index = epoch

        elapsed_time = time.time() - start_time
        if (epoch + 1) % print_interval == 0:
            print('Epoch %d Time %s' % (epoch, utils.get_time_string(elapsed_time)))
            print('T_loss %.3f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' % (
                epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
            print('V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
                val_loss, val_acc, precision, recall, fscore, mcc))

    # plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, all_fscores, all_val_fscores, cell_type)
    bst = 'Best results: %d \t V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
        best_epoch_index, best_val_loss, best_val_acc, best_val_pre, best_val_rec, best_val_fscore, best_val_mcc)
    print(bst)
    return bst


def svm_baseline(X, Y, X_test, Y_test, method=None):
    clf = SVC(gamma='auto', class_weight='balanced', probability=True).fit(X, Y)

    # Training metrics
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    # Validation metrics
    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    pd.DataFrame(Y).to_csv("/scratch/project16/TEMPO-ESM/mycode/Y_train.txt", sep="\t", index=False)
    pd.DataFrame(train_predictions).to_csv("/scratch/project16/TEMPO-ESM/mycode/Y_train_pred.txt", sep="\t",
                                           index=False)
    pd.DataFrame(Y_pred).to_csv("/scratch/project16/TEMPO-ESM/mycode/Y_test_pred.txt", sep="\t", index=False)
    pd.DataFrame(Y_test).to_csv("/scratch/project16/TEMPO-ESM/mycode/Y_test.txt", sep="\t", index=False)

    # Print metrics
    print('SVM baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    # Save results if method is specified
    if method is not None:
        with open('./reports/results/{}_SVM.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    # # Binarize the labels for ROC-AUC calculation
    # classes = clf.classes_
    # Y_test_binarized = label_binarize(Y_test, classes=classes)
    # y_test_prob = clf.predict_proba(X_test)

    # # Calculate ROC-AUC score with binarized labels
    # roc_auc = roc_auc_score(Y_test_binarized, y_test_prob, multi_class='ovr', average='micro')
    # print("ROC-AUC Score (micro average):", roc_auc)

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title("SVM ROC-AUC (Multi-Class, micro Average)")
    # plt.legend(loc='best')
    # plt.show()


def random_forest_baseline(X, Y, X_test, Y_test, method=None):
    clf = ensemble.RandomForestClassifier().fit(X, Y)
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('Random Forest baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    if method is not None:
        with open('./reports/results/{}_RF.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    # # Binarize the labels for ROC-AUC calculation
    # classes = clf.classes_
    # Y_test_binarized = label_binarize(Y_test, classes=classes)
    # y_test_prob = clf.predict_proba(X_test)

    # # Calculate ROC-AUC score with binarized labels
    # roc_auc = roc_auc_score(Y_test_binarized, y_test_prob, multi_class='ovr', average='micro')
    # print("ROC-AUC Score (micro average):", roc_auc)

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--', label='random')
    # plt.title("Random Forest ROC-AUC (Multi-Class, micro Average)")
    # plt.legend(loc='best')
    # plt.show()


def knn_baseline(X, Y, X_test, Y_test, method=None):
    clf = KNeighborsClassifier().fit(X, Y)
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('KNN baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    if method is not None:
        with open('./reports/results/{}_KNN.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    # # Binarize the labels for ROC-AUC calculation
    # classes = clf.classes_
    # Y_test_binarized = label_binarize(Y_test, classes=classes)
    # y_test_prob = clf.predict_proba(X_test)

    # # Calculate ROC-AUC score with binarized labels
    # roc_auc = roc_auc_score(Y_test_binarized, y_test_prob, multi_class='ovr', average='micro')
    # print("ROC-AUC Score (micro average):", roc_auc)

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title("KNN ROC-AUC (Multi-Class, micro Average)")
    # plt.legend(loc='best')
    # plt.show()


def bayes_baseline(X, Y, X_test, Y_test, method=None):
    clf = GaussianNB().fit(X, Y)
    train_predictions = clf.predict(X)
    train_acc = accuracy_score(Y, train_predictions)
    train_pre = precision_score(Y, train_predictions, average='micro')
    train_rec = recall_score(Y, train_predictions, average='micro')
    train_fscore = f1_score(Y, train_predictions, average='micro')
    train_mcc = matthews_corrcoef(Y, train_predictions)

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('Bayes baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    # # Binarize the labels for ROC-AUC calculation
    # classes = clf.classes_
    # Y_test_binarized = label_binarize(Y_test, classes=classes)
    # y_test_prob = clf.predict_proba(X_test)

    # # Calculate ROC-AUC score with binarized labels
    # roc_auc = roc_auc_score(Y_test_binarized, y_test_prob, multi_class='ovr', average='micro')
    # print("ROC-AUC Score (micro average):", roc_auc)

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title("Bayes ROC-AUC (Multi-Class, micro Average)")
    # plt.legend(loc='best')
    # plt.show()


# def xgboost_baseline(X, Y, X_test, Y_test, method=None):
#     clf = XGBClassifier().fit(X, Y)
#     train_acc = accuracy_score(Y, clf.predict(X))
#     train_pre = precision_score(Y, clf.predict(X))
#     train_rec = recall_score(Y, clf.predict(X))
#     train_fscore = f1_score(Y, clf.predict(X))
#     train_mcc = matthews_corrcoef(Y, clf.predict(X))
#
#     Y_pred = clf.predict(X_test)
#     precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
#     print('Logistic regression baseline:')
#     print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
#                 % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
#     print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
#                 % (val_acc, precision, recall, fscore, mcc))
#     #roc curve
#     y_pred_roc = clf.predict_proba(X_test)[:, 1]
#     fpr_rt_xgb, tpr_rt_xgb, _ = roc_curve(Y_test, y_pred_roc)
#     print(auc(fpr_rt_xgb, tpr_rt_xgb))
#     plt.figure(1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_rt_xgb, tpr_rt_xgb, label='LR')
#     plt.legend(loc='best')
#     #plt.show()


def logistic_regression_baseline(X, Y, X_test, Y_test, method=None):
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X), average='micro')
    train_rec = recall_score(Y, clf.predict(X), average='micro')
    train_fscore = f1_score(Y, clf.predict(X), average='micro')
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    val_acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='micro')
    recall = recall_score(Y_test, Y_pred, average='micro')
    fscore = f1_score(Y_test, Y_pred, average='micro')
    mcc = matthews_corrcoef(Y_test, Y_pred)

    print('Logistic regression baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    # # Binarize the labels for ROC-AUC calculation
    # classes = clf.classes_
    # Y_test_binarized = label_binarize(Y_test, classes=classes)
    # y_test_prob = clf.predict_proba(X_test)

    # # Calculate ROC-AUC score with binarized labels
    # roc_auc = roc_auc_score(Y_test_binarized, y_test_prob, multi_class='ovr', average='micro')
    # print("ROC-AUC Score (micro average):", roc_auc)

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.title("LR ROC-AUC (Multi-Class, micro Average)")
    # plt.legend(loc='best')
    # plt.show()