% SVM Program 

clear all;
close all;
clc;

% Define path for input data
path_datos='C:/.../Data/';
% Define route for collecting results
path_resultados='C:/.../Results/';

salto_frec=43.066406;

training_percent=80;
test_percent=15;
validation_percent=5;

ind_nombre=1;

%9-class     for ind_muestra=1:75 (3 impacts)       :94 (5 impacts)
% num_categorias=9;
% cell_tipos = cell(num_categorias, 4);
% cell_tipos(:,ind_nombre) = {'MgCl2_1M'; 'MgCl2_01M'; 'MgCl2_001M'; 'MgCl2_0001M'; 'KCl_1M'; 'KCl_01M'; 'KCl_001M'; 'KCl_0001M'; 'agua_pura'};

%4-class                for ind_muestra=1:78 (3 impacts)      :94 (5 impacts)
% num_categorias=4;     
% cell_tipos = cell(num_categorias, 4);
% cell_tipos(:,ind_nombre) = {'MgCl2_1M'; 'MgCl2_01M'; 'MgCl2_001M'; 'MgCl2_0001M'};

%4-class           for ind_muestra=1:75 (3 impacts)       :149 (5 impacts)
num_categorias=4;
cell_tipos = cell(num_categorias, 4);
cell_tipos(:,ind_nombre) = {'KCl_1M'; 'KCl_01M'; 'KCl_001M'; 'KCl_0001M'};

%2-class KCl             for ind_muestra=1:75 (3 impacts)     :146 (5 impacts)
% num_categorias=2;
% cell_tipos = cell(num_categorias, 4);
% cell_tipos(:,ind_nombre) = {'agua_pura';  'KCl_0001M'};

%2-class MgCl2             for ind_muestra=1:78 (3 impacts)     :146 (5 impacts)
% num_categorias=2;
% cell_tipos = cell(num_categorias, 4);
% cell_tipos(:,ind_nombre) = {'agua_pura';  'MgCl2_0001M'};



neuron_number=num_categorias;
data=[];
frecuencias=[];
gab_ind_frec=0;

nombre_base='espectro_';
for x_categoria=1:num_categorias

    for ind_muestra=1:149 
    
        nombre=strcat(nombre_base,cell_tipos{x_categoria,ind_nombre},'_',string(ind_muestra) ,'.txt');
        
        data_file=strcat(path_datos,nombre);
        fileID = fopen(data_file,'r');
        if fileID >= 3
            
            fgetl(fileID);%discard first line
         
            % Read the rest of the file as text
            rawLines = textscan(fileID, '%s', 'Delimiter', '\n');
            fclose(fileID);
            
            % Process each line: replace commas with dots and convert to numbers
            numLines = numel(rawLines{1});
            data_muestra = zeros(numLines, 2);% Preassign matrix
            
            for i = 1:numLines
                % Replace comma with decimal point
                line = strrep(rawLines{1}{i}, ',', '.');
                % Split by tabulation and convert to number
                values = sscanf(line, '%f\t%f');
                data_muestra(i,:)=values';
            end
        end
        numLines=numLines+1;
        data_muestra(numLines,1)=22049;
        data_muestra(numLines,2)=x_categoria;
        data=[data;data_muestra(:,2)'];
    
    
    end
end

frecuencias=data_muestra(:,1)';

[num_data,num_charact]=size(data);
num_charact=num_charact-1;%The last column is the field to be predicted

% normalize NOT THE LAST COLUMN
valor_maximo=max(abs(data(:,1:end-1)));
maximo=max(valor_maximo);
data(:,1:end-1)= data(:,1:end-1)/maximo;


if (training_percent + test_percent + validation_percent) > 100
   disp('Training, testing, and validation percentages with default values');
    training_percent=90;%80
    test_percent=10;
    validation_percent=0;%10
end


%Select data

max_index_data=zeros(1,3);
max_index_data(1)=floor(num_data*training_percent/100);
max_index_data(2)=max_index_data(1)+floor(num_data*test_percent/100);
max_index_data(3)=max_index_data(2)+floor(num_data*validation_percent/100);
valor=num_data - max_index_data(3);
if (valor~=0)
    max_index_data=max_index_data+valor;
end


% Add results
nombre='Iter_accuracy.txt';
data_file=strcat(path_resultados,nombre);
fid = fopen(data_file, 'a');
if fid == -1
    error('The file could not be opened.');
end


accuracy_training=[];
precision_training=[];
recall_training=[];
f1_training=[];

indicesTop=[];

for iter=1:20
    
    %RANDOM selection of data for train, test, and validation sets
    indices=randperm(num_data);
    data_train=data(indices(1:max_index_data(1)),:);
    data_test=data(indices((max_index_data(1)+1):max_index_data(2)),:);
    data_validate=data(indices((max_index_data(2)+1):end),:);
    
   
    Xtrain = data_train(:,1:end-1);
    Ytrain = data_train(:,end);
    Xtest = data_test(:,1:end-1);
    Ytest = data_test(:,end);
    Xval = data_validate(:,1:end-1);
    Yval = data_validate(:,end);
    
   
    % Training SVM
    modeloSVM = fitcecoc(Xtrain, Ytrain, ...
    'Learners', templateSVM('KernelFunction', 'linear', 'KernelScale', 'auto'), ...
    'Coding', 'onevsall');

   
    % Evaluate model
    [Ypred, scores] = predict(modeloSVM, Xtest);
    if iter<2
        % Scores
        figure;
        boxchart(scores); grid on;
        xlabel('Class', 'FontSize',18); 
        ylabel('Score', 'FontSize',18);
        title('Distribution of scores by class', 'FontSize', 20);
        ax = gca;
        ax.XAxis.FontSize = 16;
        ax.YAxis.FontSize = 16;
        
        nombre='SVM_scores_training_Id9_';
        myfile=strcat(path_resultados,nombre);
        Name=strcat(myfile,string(iter));
        Name1=strcat(Name,'.jpg');
        saveas(gcf,Name1);
    end

    accuracy = sum(Ypred == Ytest) / numel(Ytest);
    fprintf(fid, '\nIteration\t%d\nTesting dataset accuracy:\t %.2f%%\n\n', iter, accuracy * 100);

    % Evaluate (validation)  
    Ypred = predict(modeloSVM, Xval);
   
    accuracy = sum(Ypred == Yval) / numel(Yval);
    fprintf(fid, 'Validation dataset accuracy:\t %.2f%%\n', accuracy * 100);

    clases = unique(Yval);
    numClases = numel(clases);
   
    
    accuracy_iteration=[];
    precision_iteration=[];
    recall_iteration=[];
    f1_iteration=[];
    for i = 1:numClases
        clase = clases(i);
        TP = sum((Ypred == clase) & (Yval == clase));
        TN = sum((Ypred ~= clase) & (Yval ~= clase));
        FP = sum((Ypred == clase) & (Yval ~= clase));
        FN = sum((Ypred ~= clase) & (Yval == clase));
        

        accuracy_por_clase = (TP + TN) / (TP + TN + FP + FN);
        precision = TP / (TP + FP );
        recall = TP / (TP + FN );
        F1_scores = 2 * (precision * recall) / (precision + recall );

        accuracy_iteration=[accuracy_iteration accuracy_por_clase];
        precision_iteration=[precision_iteration precision];
        recall_iteration=[recall_iteration recall];
        f1_iteration=[f1_iteration F1_scores];
        
        fprintf(fid, 'Class %d\n\t Accuracy: %.2f%%\n\t Precision: %.2f%%\n\t Recall: %.2f%%\n\t F1-score: %.2f%%\n\n', ...
            clase, accuracy_por_clase*100, precision * 100, recall * 100, F1_scores * 100);
    end

    accuracy_training(iter)=mean(accuracy_iteration);
    precision_training(iter)=mean(precision_iteration);
    recall_training(iter)=mean(recall_iteration);
    f1_training(iter)=mean(f1_iteration);

  
    % Features importance 
    learner = modeloSVM.BinaryLearners{1};%first learner
    if isobject(learner) && isprop(learner,'KernelParameters')

        num_max=8;
      
        if strcmp(learner.KernelParameters.Function, 'linear')
            
            numLearners = numel(modeloSVM.BinaryLearners);
            coefMatrix = zeros(numLearners, size(Xtrain,2));
            for i = 1:numLearners
                coefMatrix(i,:) = modeloSVM.BinaryLearners{i}.Beta';
            end
            
            importancia = mean(abs(coefMatrix),1);
            min_value=min(importancia);
            importancia=importancia-min_value;

            [valoresMaximos, indicesMaximos] = maxk(importancia, num_max);
            indicesTop=[indicesTop indicesMaximos];

            fprintf(fid, '\nThe %d most important characteristics, listed in descending order, are:\n',num_max);
            for i = 1:num_max
                fprintf(fid, 'Frecuency #%d (%.0f Hz): \t%.4f\n', ...
                    (gab_ind_frec+indicesMaximos(i)), round(frecuencias(gab_ind_frec+indicesMaximos(i))), valoresMaximos(i));
            end


            % Features importance
            ancho = 2100;   % pixels
            alto = 600;    % pixels
            figure('Position', [100, 100, ancho, alto]);
            bar(importancia);

            gap_x_axis=50;
            xticks(1:gap_x_axis:num_charact); 

            idx = (gab_ind_frec+1):gap_x_axis:(gab_ind_frec+num_charact);
            xticklabels(compose('%.2f', frecuencias(idx)/1000));
            xlabel('Frequency (kHz)', 'FontWeight', 'bold', 'FontSize', 18);

            ylabel('Average importance |Beta|', 'FontWeight', 'bold', 'FontSize', 18);
            title('Significance of spectral components (linear SVM coefficients) dataset Id=10', 'FontSize', 20);

            
            ax = gca;
            ax.XAxis.FontSize = 16;
            ax.YAxis.FontSize = 16;

            for i = 1:num_max
                x = indicesMaximos(i);
                y = importancia(x);

                hold on;  
                plot(x, y, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
            end    

            frecuenciasTop = round(frecuencias(gab_ind_frec+indicesMaximos)); 

            textoCaja = 'Major frequencies:';
            for i = 1:num_max
                linea = sprintf('\n%d. %.2f kHz (#%d)', i, round(frecuenciasTop(i))/1000, (gab_ind_frec+indicesMaximos(i)));
                textoCaja = [textoCaja, linea];
            end

            annotation('textbox', [0.89, 0.73, 0.35, 0.15], ...
                'String', textoCaja, ...
                'FitBoxToText', 'on', ...
                'BackgroundColor', 'white', ...
                'EdgeColor', 'black', ...
                'FontSize', 12);

            nombre='SVM_importance_';
            myfile=strcat(path_resultados,nombre);
            Name=strcat(myfile,string(iter));
            Name1=strcat(Name,'.jpg');
            saveas(gcf,Name1);
        else
            disp('Non-linear kernel');
        end
    end

    % Confusion matrix
    fig=figure;
    cm=confusionchart(Yval, Ypred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
    cm.Title= 'SVM - Confusion Matrix for Validation dataset ';
    

    cm.DiagonalColor = [0 0.6 0];  % green color

    fig_Position = fig.Position;
    fig_Position(3) = fig_Position(3)*1.5;
    fig.Position = fig_Position;

    
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';

        annotation('textbox', [0.78, 0.16, 0.07, 0.15], ...
        'String', 'Recall ', ...
        'Color', [0 0 0], ...        % Text color: black
        'FontWeight', 'bold', ...
        'EdgeColor', 'none', ...
        'FontSize', 10);

    annotation('textbox', [0.72, 0.19, 0.3, 0.05], ...
        'String', 'Precision ', ...
        'Color', [0 0 0], ...        % Text color: black
        'FontWeight', 'bold', ...
        'EdgeColor', 'none', ...
        'FontSize', 10);

    acc_string=strcat('Accuracy=', num2str(accuracy, '%.3f'));
    annotation('textbox', [0.78, 0.12, 0.9, 0.05], ...
        'String', acc_string, ...
        'Color', [0 0 0], ...        % Text color: black
        'FontWeight', 'bold', ...
        'EdgeColor', 'none', ...
        'FontSize', 12);

    nombre='SVM_confusion_';
    myfile=strcat(path_resultados,nombre);
    Name=strcat(myfile,string(iter));
    Name1=strcat(Name,'.jpg');
    saveas(gcf,Name1);
    




end

fprintf(fid, '\n\nFINAL TRAINING ITERATIONS\nAccuracy: %.2f%%\nTRAINING Precision: %.2f%%\nTRAINING Recall: %.2f%%\nTRAINING F1-score: %.2f%%\n\n', ...
            mean(accuracy_training)*100, mean(precision_training) * 100, mean(recall_training) * 100, mean(f1_training) * 100);


accuracy_validation=[];
precision_validation=[];
recall_validation=[];
f1_validation=[];

indicesTop=unique(indicesTop);
num_features_top=numel(indicesTop);


for iter=1:150

    indices=randperm(num_data);
    data_train=data(indices(1:max_index_data(1)),:);
    data_test=data(indices((max_index_data(1)+1):max_index_data(2)),:);
    data_validate=data(indices((max_index_data(2)+1):end),:);

    Xtrain = data_train(:,1:end-1);
    Ytrain = data_train(:,end);
    
    data_test=[data_test;data_validate];
    Xtest = data_test(:,1:end-1);
    Ytest = data_test(:,end);
    
    
    Xtrain_top = Xtrain(:, indicesTop);
    Xtest_top   = Xtest(:, indicesTop);


    modeloSVM_top = fitcecoc(Xtrain_top, Ytrain, ...
    'Learners', templateSVM('KernelFunction', 'linear', 'KernelScale', 'auto'), ...
    'Coding', 'onevsall');

    
    [Ypred_val_top, scores] = predict(modeloSVM_top, Xtest_top);
    if iter<2
   
        figure;
        boxchart(scores); grid on;
        xlabel('Class', 'FontSize',18); 
        ylabel('Score', 'FontSize',18);
        title('Distribution of scores by class', 'FontSize', 20);
        ax = gca;
        ax.XAxis.FontSize = 16;
        ax.YAxis.FontSize = 16;
   
        nombre='SVM_scores_validation_Id10_';
        myfile=strcat(path_resultados,nombre);
        Name=strcat(myfile,string(iter));
        Name1=strcat(Name,'.jpg');
        saveas(gcf,Name1);
    end

    accuracy = sum(Ypred_val_top == Ytest) / numel(Ytest);
    fprintf(fid, 'Iteration\t%d\nValidation dataset accuracy:\t %.2f%%\n',iter, accuracy * 100);

    clases = unique(Ytest);
    numClases = numel(clases);
    
    accuracy_iteration=[];
    precision_iteration=[];
    recall_iteration=[];
    f1_iteration=[];
    for i = 1:numClases
        clase = clases(i);
        TP = sum((Ypred_val_top == clase) & (Ytest == clase));
        TN = sum((Ypred_val_top ~= clase) & (Ytest ~= clase));
        FP = sum((Ypred_val_top == clase) & (Ytest ~= clase));
        FN = sum((Ypred_val_top ~= clase) & (Ytest == clase));
        
        accuracy_por_clase = (TP + TN) / (TP + TN + FP + FN);
        precision = TP / (TP + FP );
        recall = TP / (TP + FN );
        F1_scores = 2 * (precision * recall) / (precision + recall );

        accuracy_iteration=[accuracy_iteration accuracy_por_clase];
        precision_iteration=[precision_iteration precision];
        recall_iteration=[recall_iteration recall];
        f1_iteration=[f1_iteration F1_scores];
        
        fprintf(fid, 'Class %d\n\t Accuracy: %.2f%%\n\t Precision: %.2f%%\n\t Recall: %.2f%%\n\t F1-score: %.2f%%\n\n', ...
            clase, accuracy_por_clase*100, precision * 100, recall * 100, F1_scores * 100);

    end

    accuracy_validation(iter)=mean(accuracy_iteration);
    precision_validation(iter)=mean(precision_iteration);
    recall_validation(iter)=mean(recall_iteration);
    f1_validation(iter)=mean(f1_iteration);

end


fprintf(fid,'\n\nFINAL accuracy with %d most important features COMBINED and %d repetitions for the VALIDATION set: %.2f%%\n', ...
    num_features_top, iter, mean(accuracy_validation) * 100);


fprintf(fid, '\n\nVALIDATION Precision: %.2f%%\nVALINDATION Recall: %.2f%%\nVALIDATION F1: %.2f%%\n\n', ...
         mean(precision_validation)*100, mean(recall_validation)*100, mean(f1_validation)*100);
  

fprintf(fid, '\nThe %d most important features are:\n',num_features_top);
for i = 1:num_features_top
    fprintf(fid, 'Frecuency #%d (%.0f Hz)\n', ...
        (gab_ind_frec+indicesTop(i)), round(frecuencias(gab_ind_frec+indicesTop(i))));
end



fclose(fid);



% Features importance

gab_ind_frec=0;
ancho = 2100;   % pixels
alto = 300;    % pixels    
figure('Position', [100, 100, ancho, alto]);
hold on
x=[1:1:num_charact];
y=ones(1,num_charact);
y=y*0.02;
b1=bar(x,y);

y=zeros(1,num_charact);
for i=1:numel(indicesTop)
    y(indicesTop(i))=1;
end
b2=bar(x,y);
b2.FaceColor='red';
hold off

gap_x_axis=12; % Range 0-20 kHz
xticks(1:gap_x_axis:num_charact);

idx = (gab_ind_frec+1):gap_x_axis:(gab_ind_frec+num_charact);
xticklabels(compose('%.1f', frecuencias(idx)/1000));
xlabel('Frequency (kHz)', 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Weight', 'FontWeight', 'bold', 'FontSize', 18);

ylim([0 1.01]);
yticks(0:0.5:1);

title('Significance spectral component (dataset Id= )', 'FontSize', 20);

ax = gca;
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
   

nombre='SVM-significance_Id10_';
myfile=strcat(path_resultados,nombre);
Name=strcat(myfile,string(iter));
Name1=strcat(Name,'.jpg');
saveas(gcf,Name1);
