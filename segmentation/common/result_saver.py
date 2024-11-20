from common.metrics import mean_dice_score, classes_IoU, classses_dice_score, mean_IoU
import json

class ResultSaver():
    def __init__(self, extra_train_variables = None, extra_val_variables = None, apply_softmax = True) -> None:
        self.apply_softmax = apply_softmax
        self.results = {
            "train_losses": [],
            "val_losses": [],
            "val_ious_classes": [],
            "val_dices_classes": [],
            "val_ious_mean": [],
            "val_dices_mean": []
        }
        self.epoch_results = {
            "train_losses": [],
            "val_losses": [],
            "val_ious_classes": [],
            "val_dices_classes": [],
            "val_ious_mean": [],
            "val_dices_mean": []
        }


        self.extra_train_variables = extra_train_variables
        if extra_train_variables:
            for variable in extra_train_variables:
                self.results[variable] = []
                self.epoch_results[variable] = []

        self.extra_val_variables = extra_val_variables
        if extra_val_variables:
            for variable in extra_val_variables:
                self.results[variable] = []
                self.epoch_results[variable] = []

    def mean(self, l):
        return sum(l)/len(l)


    def train_step(self, loss):
        if self.extra_train_variables:
            self.epoch_results["train_losses"].append(loss[0].item())
            for i, extra_train_variable in enumerate(self.extra_train_variables):
                self.epoch_results[extra_train_variable].append(loss[i+1].item())
        else:
            self.epoch_results["train_losses"].append(loss.item())

    def val_step(self, loss, predicted, label):
        if self.extra_val_variables:
            self.epoch_results["val_losses"].append(loss[0].item())
            for i, extra_val_variable in enumerate(self.extra_val_variables):
                self.epoch_results[extra_val_variable].append(loss[i+1].item())
        else:
            self.epoch_results["val_losses"].append(loss.item())
        
        ious = classes_IoU(predicted, label)
        self.epoch_results["val_ious_classes"].append(ious)
        self.epoch_results["val_ious_mean"].append(self.mean(ious))

        dices = classses_dice_score(predicted, label, self.apply_softmax)
        self.epoch_results["val_dices_classes"].append(dices)
        self.epoch_results["val_dices_mean"].append(self.mean(dices))

    def val_step_no_computation(self, loss, dice, iou):
        if self.extra_val_variables:
            self.epoch_results["val_losses"].append(loss[0].item())
            for i, extra_val_variable in enumerate(self.extra_val_variables):
                self.epoch_results[extra_val_variable].append(loss[i+1].item())
        else:
            self.epoch_results["val_losses"].append(loss.item())
        
        self.epoch_results["val_ious_classes"].append(iou)
        self.epoch_results["val_ious_mean"].append(self.mean(iou))

        self.epoch_results["val_dices_classes"].append(dice)
        self.epoch_results["val_dices_mean"].append(self.mean(dice))
        

    def epoch_step(self, print_score_short = True, print_all_score = False):
        
        # print(self.epoch_results)
        for key, value in self.epoch_results.items():
            if isinstance(value[0], list):
                means = [sum(values)/ len(values) for values in zip(*value)]
                self.results[key].append(means)
            else: 
                self.results[key].append(self.mean(value))

        for key in self.epoch_results.keys():
            self.epoch_results[key] = []
        
        if print_score_short and not print_all_score:
            print(f'Train Loss: {self.results["train_losses"][-1]}')
            print(f'Val Loss: {self.results["val_losses"][-1]}')
            print(f'Dice: {self.results["val_dices_mean"][-1]}')
            print(f'IoU: {self.results["val_ious_mean"][-1]}')
            # print(f'Dices: {self.results["val_dices_classes"][-1]}')
            # print(f'IoUs: {self.results["val_ious_classes"][-1]}')
        if print_all_score:
            for key in self.results.keys():
                if key == "val_dices_classes" or key == "val_ious_classes":
                    pass
                else:
                    print(f'{key} : {self.results[key][-1]}')

    def print_last_value(self, key):
        print(f'{key}: {self.results[key][-1]}')
    
    def save_result(self, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(self.results, json_file)
        
    def get_val_loss(self, custom_loss_name = False):
        if custom_loss_name:
            return self.results[custom_loss_name][-1]
        else:
            return self.results["val_losses"][-1]