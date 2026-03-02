% RF Program 

clear all;
close all;
clc;

% Define path for input data
path_datos='C:/.../Data/';
% Define route for collecting results
path_resultados='C:/.../Results/';


salto_frec=43.066406;

training_percent=85;
test_percent=10;
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

    for ind_muestra=1:75      
                
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
nombre='Max_iter.txt';
data_file=strcat(path_resultados,nombre);
fid = fopen(data_file, 'a');
if fid == -1
    error('The file could not be opened.');
end
fprintf(fid, 'TRAINING\n');
    
accuracy_iter = [];
precision_iter = [];
recall_iter = [];
F1_scores_iter = [];

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
    
    data_test=[data_test;data_validate];
    Xtest = data_test(:,1:end-1);
    Ytest = data_test(:,end);
    
    
    % Train Random Forest
    Num_trees=50;
    modeloRF = TreeBagger(Num_trees, Xtrain, Ytrain, ...
        'Method', 'classification', ...
        'OOBPrediction', 'On', ...
        'OOBPredictorImportance', 'on', ...  
        'NumPredictorsToSample', round(sqrt(num_charact)), ... 
        'MinLeafSize', 5);

if iter<2
    figure;
    plot(oobError(modeloRF), 'LineWidth', 4);
    set(gca, 'FontSize', 16);  
    title('Out-of-bag error vs Number of trees', 'FontSize', 20);
    xlabel('Number of trees', 'FontSize', 18);
    ylabel('OOB error', 'FontSize', 18);
    grid on;
    
    nombre='OOB_error_';
    myfile=strcat(path_resultados,nombre);
    Name=strcat(myfile,string(iter));
    Name1=strcat(Name,'.jpg');
    saveas(gcf,Name1);
end


    [Ypred, scores] = predict(modeloRF, Xtest);
    Ypred = str2double(Ypred);    
    
    % Performance metrics
    accuracy = sum(Ypred == Ytest) / numel(Ytest);
    fprintf(fid, '\nIteration %d\nNumber of features = %d\nINITIAL accuracy of the model on the training set:\t %.2f%% (correct /num_element)\n',iter,num_charact, accuracy * 100);
    
    % Metrics by class
    clases = unique(Ytest);
    numClases = numel(clases);
        
    TP_iter=0;
    TN_iter=0;
    FP_iter=0;
    FN_iter=0;
    for i = 1:numClases
        clase = clases(i);
        TP = sum((Ypred == clase) & (Ytest == clase));
        TN = sum((Ypred ~= clase) & (Ytest ~= clase));
        FP = sum((Ypred == clase) & (Ytest ~= clase));
        FN = sum((Ypred ~= clase) & (Ytest == clase));

        TP_iter=TP_iter+ TP;
        TN_iter=TN_iter+ TN;
        FP_iter=FP_iter+ FP;
        FN_iter=FN_iter+ FN;

        
        accuracy_por_clase = (TP + TN) / (TP + TN + FP + FN);
        precision = TP / (TP + FP );
        recall = TP / (TP + FN );
        F1_scores = 2 * (precision * recall) / (precision + recall );

        accuracy_training=[accuracy_training accuracy_por_clase];
        precision_training=[precision_training precision];
        recall_training=[recall_training recall];
        f1_training=[f1_training F1_scores];
        
        fprintf(fid, 'Class %d\n\t Accuracy: %.2f%%\n\t Precision: %.2f%%\n\t Recall: %.2f%%\n\t F1-score: %.2f%%\n\n', ...
            clase, accuracy_por_clase *100, precision * 100, recall * 100, F1_scores * 100);
    end  
    accuracy_iter =[accuracy_iter  (TP_iter + TN_iter) / (TP_iter + TN_iter + FP_iter + FN_iter) ];
    prec=(TP_iter / (TP_iter + FP_iter));
    precision_iter = [precision_iter prec ];
    rec=TP_iter / (TP_iter + FN_iter);
    recall_iter = [recall_iter rec ];
    F1_scores_iter =[F1_scores_iter ( 2 * (prec * rec) / (prec + rec) ) ];

    

    % Get the importance vector
    importancia = modeloRF.OOBPermutedPredictorDeltaError;
    
    [importanciaOrdenada, idxOrden] = sort(importancia, 'descend');
    
    importanciaAcumulada = cumsum(importanciaOrdenada) / sum(importanciaOrdenada);
    
    umbral = 0.85;
    numCaracteristicas = find(importanciaAcumulada >= umbral, 1);
    
    % Finding values and indices
    num_max_1=3;
    
    umbral_1=importanciaAcumulada(num_max_1);
    numCaracteristicas_1=num_max_1;

    num_max=8;
    
    umbral_2=importanciaAcumulada(num_max);
    numCaracteristicas_2=num_max;

    fprintf(fid,'\nNumber of features needed to cover the %.0f%% of importance: %d\n', umbral*100, numCaracteristicas);
    fprintf(fid,'\nNumber of features needed to cover the %.0f%% of importance: %d\n', umbral_2*100, numCaracteristicas_2);
    fprintf(fid,'\nNumber of features needed to cover the %.0f%% of importance: %d\n', umbral_1*100, numCaracteristicas_1);

    
    caracteristicasSeleccionadas = idxOrden(1:numCaracteristicas);
    fprintf(fid,'\nIndex of selected features:\n');
    for i = 1:numel(caracteristicasSeleccionadas)
       fprintf(fid, '%d\t',caracteristicasSeleccionadas(i));
    end

    figure;
    plot(importanciaAcumulada, 'LineWidth', 3);
    xlim([1 numel(importanciaAcumulada)+10]);  
    hold on;
    yline(umbral, '--r', sprintf('%.0f%%\t(%d features)', umbral*100, numCaracteristicas), 'FontSize', 16);
    xline(numCaracteristicas, '--r', 'FontSize', 16);
    
    yline(umbral_1, '--r', sprintf('%.0f%%\t(%d features)', umbral_1*100, numCaracteristicas_1), 'FontSize', 16);
    xline(numCaracteristicas_1, '--r', 'FontSize', 16);
    
    yline(umbral_2, '--r', sprintf('%.0f%%\t(%d features)', umbral_2*100, numCaracteristicas_2), 'FontSize', 16);
    xline(numCaracteristicas_2, '--r', 'FontSize', 16);
    
    xlabel('Features number', 'FontSize', 18);
    ylabel('Cumulative significance', 'FontSize', 18);
    title('Feature selection according to significance OOB (Dataset Id=9)', 'FontSize', 20);
    set(gca, 'FontSize', 16);   
    grid on;

    nombre='Accumulated_importance_';
    myfile=strcat(path_resultados,nombre);
    Name=strcat(myfile,string(iter));
    Name1=strcat(Name,'.jpg');
    saveas(gcf,Name1);



    [valoresMaximos, indicesMaximos] = maxk(importancia, num_max);
    indicesTop=[indicesTop indicesMaximos];


    fprintf(fid, '\n\nThe %d most important characteristics, listed in descending order, are:\n',num_max);
    for i = 1:num_max
        fprintf(fid, 'Frequency #%d (%.0f Hz): \t%.4f\n', ...
            (gab_ind_frec+indicesMaximos(i)), round(frecuencias(gab_ind_frec+indicesMaximos(i))), valoresMaximos(i));
    end

    % Reduce datasets to selected features
    Xtrain_top = Xtrain(:, indicesMaximos);
    Xtest_top   = Xtest(:, indicesMaximos);

    
    modeloRF_top = TreeBagger(Num_trees, Xtrain_top, Ytrain, ...
        'Method', 'classification', ...
        'OOBPrediction', 'On', ...
        'NumPredictorsToSample', round(sqrt(num_max)), ... %'all' ...
        'MinLeafSize', 5);

  
    % Evaluate the new model in validation
    Ypred_val_top = predict(modeloRF_top, Xtest_top);
    Ypred_val_top = str2double(Ypred_val_top);
    accuracy_top = sum(Ypred_val_top == Ytest) / numel(Ytest);
    fprintf(fid,'\nAccuracy with %d most important features on training set: %.2f%% (correct/num_elemt)\n', ...
        numel(valoresMaximos), accuracy_top * 100);

end


fprintf(fid, '\n\nTRAINING average iterations \nAccuracy: %.2f%%\nPrecision: %.2f%%\nRecall: %.2f%%\nF1: %.2f%%\n\n', ...
         mean(accuracy_iter)*100,mean(precision_iter)*100, mean(recall_iter)*100, mean(F1_scores_iter)*100);


fprintf(fid, '\nArray length = %d\n',length(accuracy_iter));


fprintf(fid, '\n\nTRAINING (average classes*iterations) \nAccuracy: %.2f%%\nPrecision: %.2f%%\nRecall: %.2f%%\nF1: %.2f%%\n\n\n', ...
         mean(accuracy_training)*100,mean(precision_training)*100, mean(recall_training)*100, mean(f1_training)*100);
 

fprintf(fid, '\nArray length = %d\n',length(accuracy_training));



accuracy_validation=[];
precision_validation=[];
recall_validation=[];
f1_validation=[];




%update indices
indicesTop=unique(indicesTop);
num_features_top=numel(indicesTop);

accuracy_iter = [];
precision_iter = [];
recall_iter = [];
F1_scores_iter = [];

for iter=1:150

    %RANDOM selection of data for train, test, and validation sets
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
        
    modeloRF_top = TreeBagger(Num_trees, Xtrain_top, Ytrain, ...
        'Method', 'classification', ...
        'OOBPrediction', 'On', ...
        'NumPredictorsToSample', round(sqrt(length(indicesTop))), ... %'all'
        'MinLeafSize', 5);
    
    % Evaluate the new model in validation
    Ypred_val_top = predict(modeloRF_top, Xtest_top);
    Ypred_val_top = str2double(Ypred_val_top);
    accuracy = sum(Ypred_val_top == Ytest) / numel(Ytest); 
    
    fprintf(fid, '\n\nVALIDACION\n\nIteration %d\nFINAL accuracy of the model on the VALIDATION set:\t %.2f%% (correct/num_element)\n', iter, accuracy * 100);
    
    % metrics by class
    clases = unique(Ytest);
    numClases = numel(clases);
        
    accuracy_iteration=[];
    precision_iteration=[];
    recall_iteration=[];
    f1_iteration=[];

    TP_iter=0;
    TN_iter=0;
    FP_iter=0;
    FN_iter=0;
    for i = 1:numClases
        clase = clases(i);
        TP = sum((Ypred_val_top == clase) & (Ytest == clase));
        TN = sum((Ypred_val_top ~= clase) & (Ytest ~= clase));
        FP = sum((Ypred_val_top == clase) & (Ytest ~= clase));
        FN = sum((Ypred_val_top ~= clase) & (Ytest == clase));
        
        TP_iter=TP_iter+ TP;
        TN_iter=TN_iter+ TN;
        FP_iter=FP_iter+ FP;
        FN_iter=FN_iter+ FN;        
        
        accuracy_por_clase = (TP + TN) / (TP + TN + FP + FN);
        precision = TP / (TP + FP );
        recall = TP / (TP + FN );
        F1_scores = 2 * (precision * recall) / (precision + recall );

        accuracy_iteration=[accuracy_iteration accuracy_por_clase];
        precision_iteration=[precision_iteration precision];
        recall_iteration=[recall_iteration recall];
        f1_iteration=[f1_iteration F1_scores];
        
        fprintf(fid, 'Class %d\n\t Accuracy: %.2f%%\n\t Precision: %.2f%%\n\t Recall: %.2f%%\n\t F1-score: %.2f%%\n\n', ...
            clase, accuracy_por_clase *100, precision * 100, recall * 100, F1_scores * 100);
    end  
    accuracy_iter =[accuracy_iter  (TP_iter + TN_iter) / (TP_iter + TN_iter + FP_iter + FN_iter) ];
    prec=(TP_iter / (TP_iter + FP_iter));
    precision_iter = [precision_iter prec ];
    rec=TP_iter / (TP_iter + FN_iter);
    recall_iter = [recall_iter rec ];
    F1_scores_iter =[F1_scores_iter ( 2 * (prec * rec) / (prec + rec) ) ];



    fprintf(fid, '\nMetrics for iteration %d (average classes)\nAccuracy: %.2f%%\nPrecision: %.2f%%\nRecall: %.2f%%\nF1: %.2f%%\n\n', ...
         iter,mean(accuracy_iteration) * 100, mean(precision_iteration)*100, mean(recall_iteration)*100, mean(f1_iteration)*100);

    accuracy_validation(iter)=mean(accuracy_iteration);
    precision_validation(iter)=mean(precision_iteration);
    recall_validation(iter)=mean(recall_iteration);
    f1_validation(iter)=mean(f1_iteration);

    
end

fprintf(fid,'\n\nFINAL VALIDATION \n\n%d most important features COMBINED \n%d repetitions for the VALIDATION set:\n', ...
    num_features_top, iter);

fprintf(fid, '\n\nVALIDATION (average iterations) \nAccuracy: %.2f%%\nPrecision: %.2f%%\nRecall: %.2f%%\nF1: %.2f%%\n\n', ...
         mean(accuracy_iter)*100,mean(precision_iter)*100, mean(recall_iter)*100, mean(F1_scores_iter)*100);
fprintf(fid, '\nLongitud del array = %d\n',length(accuracy_iter));

fprintf(fid, '\nVALIDATION (average classes and iterations) \nAccuracy: %.2f%%\nPrecision: %.2f%%\nRecall: %.2f%%\nF1: %.2f%%\n\n', ...
         mean(accuracy_validation) * 100, mean(precision_validation)*100, mean(recall_validation)*100, mean(f1_validation)*100);


fprintf(fid, '\nThe %d most important features are:\n',num_features_top);
for i = 1:num_features_top
    fprintf(fid, 'Frequency #%d (%.0f Hz)\n', ...
        (gab_ind_frec+indicesTop(i)), round(frecuencias(gab_ind_frec+indicesTop(i))));
end


fclose(fid);


% Visualize importance of features

gab_ind_frec=0;
ancho = 2100;   
alto = 300;    
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



gap_x_axis=12;

xticks(1:gap_x_axis:num_charact); 


% indexes for the tags
idx = (gab_ind_frec+1):gap_x_axis:(gab_ind_frec+num_charact);


xticklabels(compose('%.1f', frecuencias(idx)/1000));% tags in kHz
xlabel('Frequency (kHz)', 'FontWeight', 'bold', 'FontSize', 18);


ylabel('Weight', 'FontWeight', 'bold', 'FontSize', 18);

ylim([0 1.01]);
yticks(0:0.5:1);

title('Significance spectral component (dataset Id=10)', 'FontSize', 20);


% Adjust font size of axis labels (ticks)
ax = gca;
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
   

nombre='RF_importance_Id10_';
myfile=strcat(path_resultados,nombre);
Name=strcat(myfile,string(iter));
Name1=strcat(Name,'.jpg');
saveas(gcf,Name1);
