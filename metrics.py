import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


class Evaluate():
    def __init__(self,save_path=None):
        self.target = None
        self.output = None
        '''
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.threshold_confusion = 0.5
        '''

    # Add data pair (target and predicted value)
    def add_batch(self,batch_tar,batch_out):
        batch_tar = batch_tar.flatten()
        batch_out = batch_out.flatten()

        self.target = batch_tar if self.target is None else np.concatenate((self.target,batch_tar))
        self.output = batch_out if self.output is None else np.concatenate((self.output,batch_out))

    def confusion_matrix(self):
        #Confusion matrix
        confusion = confusion_matrix(self.target, self.output)
        print(confusion)
        total = confusion.sum()
        total_column = confusion.sum(axis=0)
        total_row = confusion.sum(axis=1)

        self.TP = np.diagonal(confusion)

        self.FP = total_column.sum() - self.TP
            
        self.TN = total - total_column - total_row + self.TP

        self.FN = total_row.sum() - self.TP

        accuracy = 0
        if total != 0:
            accuracy = self.TP.sum() / total
        print('accuracy:', accuracy)

        precisions = 0
        precision = 0
        if total_column.all():
            precisions = self.TP / total_column
            precision = precisions.mean()
        print('precision:', precision)

        recalls = 0
        recall = 0
        if total_row.all():
            recalls = self.TP / total_row
            recall = recalls.mean()
        print('recall:', recall)
        
        F1_score = 0
        if (precisions + recalls).all():
            F1_scores = 2 * precisions * recalls / (precisions + recalls)
            F1_score = F1_scores.mean()
        print('Macro-F1:', F1_score)

        self.specificity = 0
        if (self.FP + self.TN).all():
            self.specificity = self.TN / (self.FP + self.TN)
        print('TNR:', self.specificity)

        self.FPR = 1 - self.specificity
        print('FPR:', self.FPR)
    
        return confusion, accuracy, precision, recall, F1_score
    
    
    def auc_roc(self, plot=True):
        AUC_ROC = roc_auc_score(self.target, self.output)
        # print("\nAUC of ROC curve: " + str(AUC_ROC))
        if plot:
            fpr, tpr, thresholds = roc_curve(self.target, self.output)
            # print("\nArea under the ROC curve: " + str(AUC_ROC))
            plt.figure()
            plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
            plt.title('ROC curve')
            plt.xlabel("FPR (False Positive Rate)")
            plt.ylabel("TPR (True Positive Rate)")
            plt.legend(loc="lower right")
            plt.savefig(("ROC.png"))
        return AUC_ROC

if __name__ == '__main__':  
    a = np.random.randint(0, 10, (3, 3))
    print(a)
    total = a.sum()
    row = a.sum(axis=0)
    c = a.sum(axis=1)
    d = np.diagonal(a)
    d = np.array([1, 0, 1])
    print(total)
    print('row:', row)
    print('c:', c)
    print(d)
    print(total - row - c + d)
    print(d.mean())