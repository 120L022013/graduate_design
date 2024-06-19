from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score
import seaborn as sns

def result_test(model_name,real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    recall = recall_score(real, pred)
    f1 = acc*recall*2/(acc+recall)
    patten = 'acc: %.4f   recall: %.4f   f1: %.4f'
    # print(patten % (acc, recall, f1,))
    labels11 = ['negative', 'active']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='')
    plt.savefig("results/reConfusionMatrix_"+model_name+".tif", dpi=400)
    plt.close()

def plot_acc(model_name,train_acc,dev_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    if len(train_acc)==len(dev_acc):
        y = list(range(len(train_acc)))
        plt.plot(y, dev_acc, alpha=0.9, linewidth=2, label='dev acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.text(0.5, 1.05, 'Acc Scores for ' + model_name, horizontalalignment='center', fontsize=14,
             transform=plt.gca().transAxes)
    plt.savefig("results/acc_"+model_name+".png", dpi=400)
    plt.close()

def plot_F1(model_name,train_F1,dev_F1):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_F1)))
    plt.plot(x, train_F1, alpha=0.9, linewidth=2, label='train F1')
    if len(train_F1)==len(dev_F1):
        y = list(range(len(train_F1)))
        plt.plot(y, dev_F1, alpha=0.9, linewidth=2, label='dev F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend(loc='best')
    plt.text(0.5, 1.05, 'F1 Scores for ' + model_name, horizontalalignment='center', fontsize=14,
             transform=plt.gca().transAxes)
    plt.savefig("results/F1_"+model_name+".png", dpi=400)
    plt.close()

def plot_loss(model_name,train_loss,dev_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))

    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
    if len(train_loss) == len(dev_loss):
        y = list(range(len(dev_loss)))
        plt.plot(y, dev_loss, alpha=0.9, linewidth=2, label='dev loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.text(0.5, 1.05, 'loss for ' + model_name, horizontalalignment='center', fontsize=14,
             transform=plt.gca().transAxes)
    plt.savefig('results/loss_'+model_name+'.png', dpi=400)
    plt.close()