import torch

def logging(log_name ,str_lists):
    with open(log_name, 'w') as f:  # 设置文件对象
        f.writelines(str_lists)


class bestSave:
    def __init__(self):
        self.best_score = None
    def stop_step(self, acc, model, args):
        score = acc
        if self.best_score is None:
            self.best_score = score
        elif self.best_score < score:
            self.best_score = score
            torch.save({'model_state_dict': model.state_dict()}, args.dataset + "+DGNN+V2.pt")
