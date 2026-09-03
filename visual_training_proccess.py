import matplotlib.pyplot as plt

def train_process_picture(epoch_list, train_loss_list, val_loss_list, val_acc_list, grad_norm_list):

    fig, ax1 = plt.subplots()
    ax1.plot(epoch_list, train_loss_list, 'b-', label='Train Loss')
    ax1.plot(epoch_list, val_loss_list, 'g-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(epoch_list, val_acc_list, 'r-', label='Val Acc')
    ax2.plot(epoch_list, grad_norm_list, 'm--', label='Grad Norm')  # 新增梯度范数曲线
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.legend(loc='upper right')
    plt.show()