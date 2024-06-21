
clear;
close all;

load('./confusion_matrix_SOC10.mat')

num = length(unique(label));
label = label+1;
target = full(ind2vec(double(label), num));

num = length(unique(label_predict));
label_predict = label_predict+1;
output = full(ind2vec(double(label_predict), num));
figure
plotconfusion(target,output,'SOC-10')
img = gcf;
print(img, '-dpng', '-r500', './ConfusionMatrix_SOC10.png');



load('./confusion_matrix_EOC7.mat')

num = length(unique(label));
label = label+1;
target = full(ind2vec(double(label), num));

num = length(unique(label_predict));
label_predict = label_predict+1;
output = full(ind2vec(double(label_predict), num));
figure
plotconfusion(target,output,'EOC-7')

img = gcf;
print(img, '-dpng', '-r500', './ConfusionMatrix_EOC7.png');


c = confusionmat(label, label_predict);



