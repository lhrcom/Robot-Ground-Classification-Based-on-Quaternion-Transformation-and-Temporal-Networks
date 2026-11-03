"""训练曲线：保存图片默认不弹窗，梯度范数单独显示。"""

from pathlib import Path


def train_process_picture(epoch_list, train_loss_list, val_loss_list, val_acc_list,
                          grad_norm_list, output_path=None, show=True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epoch_list, train_loss_list, label="Train")
    axes[0].plot(epoch_list, val_loss_list, label="Validation")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epoch_list, val_acc_list)
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[2].plot(epoch_list, grad_norm_list)
    axes[2].set_ylabel("Gradient L2 norm (before clipping)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(Path(output_path), dpi=160)
    if show:
        plt.show()
    plt.close(fig)
