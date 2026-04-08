import os
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0,
                 path='checkpoint.pt', trace_func=print, stop_order='min',
                 save_full_model=True, save_state_dict=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_order = stop_order
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.save_full_model = save_full_model
        self.save_state_dict = save_state_dict

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.stop_order == 'min':
            # 目标：val_loss 越小越好
            improved = (self.best_score is None) or (score > self.best_score + self.delta)
        else:
            # 目标：val_loss 越大越好（你这里用的是 rho，要最大化）
            improved = (self.best_score is None) or (score < self.best_score - self.delta)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return

        if improved:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Updation changed ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        # 统一把目录建好
        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None

        # 1) 保存权重（推荐长期使用）
        if self.save_state_dict:
            sd_path = self.path
            # 如果你传的是 predictor.pth，建议再额外生成 predictor_state_dict.pth
            if sd_path.endswith(".pth"):
                sd_path = sd_path.replace(".pth", "_state_dict.pth")
            else:
                sd_path = sd_path + "_state_dict.pth"
            torch.save(model.state_dict(), sd_path)

        # 2) 保存整模型（方便一键 load，但对版本/安全更敏感）
        if self.save_full_model:
            full_path = self.path
            if full_path.endswith(".pth"):
                full_path = full_path.replace(".pth", "_full_model.pth")
            else:
                full_path = full_path + "_full_model.pth"
            torch.save(model, full_path)

        self.val_loss_min = val_loss
