% GA + ELM 

clear all; close all; clc;


path_datos      = 'C:/.../Data/';
path_resultados = 'C:/.../Results/';


rng(260101);                       % Reproducible seed

time_limit_minutes = 30;         % Time limit per execution
time_limit_seconds = time_limit_minutes*60;

salto_frec = 43.066406;         


training_percent   = 85;
test_percent       = 10;
validation_percent = 5;

ind_nombre = 1;




% 9 classes 
%       ( ind_muestra ~ 1:75 (3 impacts) o 1:94 (5 impacts))
num_categorias = 9;
cell_tipos = cell(num_categorias, 4);
cell_tipos(:, ind_nombre) = { ...
    'MgCl2_1M'; 'MgCl2_01M'; 'MgCl2_001M'; 'MgCl2_0001M'; ...
    'KCl_1M';   'KCl_01M';   'KCl_001M';   'KCl_0001M';   'agua_pura' ...
    };
num_muestras_por_clase = 75;



data = [];
frecuencias = [];
gab_ind_frec = 0;

nombre_base = 'espectro_';
for x_categoria = 1:num_categorias
    for ind_muestra = 1:num_muestras_por_clase
        nombre = strcat(nombre_base, cell_tipos{x_categoria, ind_nombre}, '_', string(ind_muestra), '.txt');
        data_file = fullfile(path_datos, nombre);
        fileID = fopen(data_file, 'r');
        if fileID >= 3
            fgetl(fileID); %discard first line
            rawLines = textscan(fileID, '%s', 'Delimiter', '\n');
            fclose(fileID);

            numLines = numel(rawLines{1});
            data_muestra = zeros(numLines, 2);
            for i = 1:numLines
                line = strrep(rawLines{1}{i}, ',', '.');
                values = sscanf(line, '%f\t%f');
                data_muestra(i,:) = values';
            end
        else
            error('It could not be opened: %s', data_file);
        end
        numLines = numLines + 1;
        data_muestra(numLines, 1) = 22049;      % end
        data_muestra(numLines, 2) = x_categoria;% label
        data = [data; data_muestra(:,2)'];      % a row per sample: [features ... label]
    end
end

frecuencias = data_muestra(:,1)';

[num_data, num_charact] = size(data);
num_charact = num_charact - 1;  %The last column is the field to be predicted

% Normalize
valor_maximo = max(abs(data(:,1:end-1)));
maximo = max(valor_maximo);
if maximo > 0
    data(:,1:end-1) = data(:,1:end-1) / maximo;
end


if (training_percent + test_percent + validation_percent) ~= 100
    error('Training, testing, and validation percentages with default values.');
end


% Intervals (subsets) 2-3-4-5
interval_lengths = [2 3 4 5];
subsets = build_subsets(num_charact, interval_lengths);   % indexes
num_subsets = numel(subsets);

% Chromosome-coded group parameters
G = 10;                              % number of simultaneous groups 
min_features_per_group = num_categorias;   % threshold to avoid wrong solutions


% Parameters ELM 
Elm_Type = 1;                        % classes
ActivationFunction = 'sig';
NumberofHiddenNeurons = max(20, 5*num_categorias);
kfold = 5;                           % Internal cross valid  (training)
alpha_acc = 1.0;                     
beta_size  = 0.01;                   % relative size penalty

elmParams.Elm_Type = Elm_Type;
elmParams.ActivationFunction = ActivationFunction;
elmParams.NumberofHiddenNeurons = NumberofHiddenNeurons;

% GA
gaParams.popSize    = 60;
gaParams.maxGens    = 60;
gaParams.tournament = 3;
gaParams.Pc         = 0.8;                % recombination
gaParams.Pm         = 1/num_subsets;      % mutation
gaParams.G          = G;
gaParams.minFeat    = min_features_per_group;
gaParams.beta       = beta_size;
gaParams.alpha      = alpha_acc;
gaParams.kfold      = kfold;
gaParams.m          = num_charact;
gaParams.elm        = elmParams;



max_index_data = zeros(1,3);
max_index_data(1) = floor(num_data * training_percent   / 100);
max_index_data(2) = max_index_data(1) + floor(num_data * test_percent / 100);
max_index_data(3) = max_index_data(2) + floor(num_data * validation_percent / 100);
valor = num_data - max_index_data(3);
if valor ~= 0
    max_index_data = max_index_data + valor;
end



nombre = 'GA_ELM_results.txt';
data_file = fullfile(path_resultados, nombre);
fid = fopen(data_file, 'a');
if fid == -1, error('The file could not be opened.'); end

fprintf(fid, '\n\n GA + ELM  \n\n');
fprintf(fid, 'Seed RNG | Partitions = %d/%d/%d | N1=%d | N2=%d | G=%d\n', ...
    training_percent, test_percent, validation_percent, 20, 150, G);
fprintf(fid, 'ELM: H=%d | act=%s | kfold=%d | fitness = accuracy - %.4f*(|S|/m)\n', ...
    NumberofHiddenNeurons, ActivationFunction, kfold, beta_size);



% Block 1
N1 = 20;

accuracy_iter = [];
precision_iter = [];
recall_iter = [];
F1_scores_iter = [];

accuracy_training = [];
precision_training = [];
recall_training = [];
f1_training = [];


selection_frequency = zeros(1, num_charact);

for iter = 1:N1
    tic_iter = tic;

    
    indices = randperm(num_data);
    data_train    = data(indices(1:max_index_data(1)), :);
    data_test     = data(indices((max_index_data(1)+1):max_index_data(2)), :);
    data_validate = data(indices((max_index_data(2)+1):end), :);

    % Evaluation
    data_eval = [data_test; data_validate];

    % k-fold in TRAIN 
    folds = build_stratified_kfolds(data_train(:,end), kfold);

    
    deadline = tic;
    [bestMask, bestGroup, bestAccCV, bestSize] = run_ga_feature_selection( ...
        data_train, subsets, gaParams, folds, time_limit_seconds, deadline);

    % 
    selection_frequency = selection_frequency + double(bestMask);

    % Evaluate ELM 
    selected_idx = find(bestMask);
    if isempty(selected_idx)
        warning('Iter %d: empty subset; skips external evaluation.', iter);
        continue;
    end

    [Ypred_eval, TestingAccuracy] = eval_ELM_once(data_train, data_eval, selected_idx, elmParams);

    Ytrue_eval = data_eval(:, end);
    accuracy = TestingAccuracy;  

    % Metrics (macro)
    clases = unique(Ytrue_eval);
    numClases = numel(clases);

    accuracy_iteration = [];
    precision_iteration = [];
    recall_iteration = [];
    f1_iteration = [];

    TP_iter=0; TN_iter=0; FP_iter=0; FN_iter=0;

    for iC = 1:numClases
        clase = clases(iC);
        TP = sum((Ypred_eval == clase) & (Ytrue_eval == clase));
        TN = sum((Ypred_eval ~= clase) & (Ytrue_eval ~= clase));
        FP = sum((Ypred_eval == clase) & (Ytrue_eval ~= clase));
        FN = sum((Ypred_eval ~= clase) & (Ytrue_eval == clase));

        TP_iter=TP_iter+TP; TN_iter=TN_iter+TN; FP_iter=FP_iter+FP; FN_iter=FN_iter+FN;

        acc_c = (TP+TN) / (TP+TN+FP+FN + eps);
        prec  = TP / (TP + FP + eps);
        rec   = TP / (TP + FN + eps);
        F1    = 2*(prec*rec)/(prec+rec + eps);

        accuracy_iteration = [accuracy_iteration acc_c];
        precision_iteration = [precision_iteration prec];
        recall_iteration = [recall_iteration rec];
        f1_iteration = [f1_iteration F1];
    end

    acc_global = (TP_iter + TN_iter) / (TP_iter + TN_iter + FP_iter + FN_iter + eps);
    prec_global = TP_iter / (TP_iter + FP_iter + eps);
    rec_global  = TP_iter / (TP_iter + FN_iter + eps);
    F1_global   = 2*(prec_global*rec_global)/(prec_global+rec_global + eps);

    accuracy_iter    = [accuracy_iter acc_global];
    precision_iter   = [precision_iter prec_global];
    recall_iter      = [recall_iter rec_global];
    F1_scores_iter   = [F1_scores_iter F1_global];

    accuracy_training(iter) = mean(accuracy_iteration);
    precision_training(iter)= mean(precision_iteration);
    recall_training(iter)   = mean(recall_iteration);
    f1_training(iter)       = mean(f1_iteration);

    
    fprintf(fid, 'ITER %d  | bestAcc_CV=%.4f | |S|=%d | ext-AC=%.4f | time=%.1fs\n', ...
        iter, bestAccCV, bestSize, accuracy, toc(tic_iter));
    fprintf(fid, 'Acc: %.2f%%  Prec: %.2f%%  Rec: %.2f%%  F1: %.2f%%\n\n', ...
        mean(accuracy_iteration)*100, mean(precision_iteration)*100, mean(recall_iteration)*100, mean(f1_iteration)*100);

    % Confusion matrix
    fig = figure('Visible','off');
    cm = confusionchart(Ytrue_eval, Ypred_eval, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
    cm.Title = sprintf('GA+ELM - Confusion Matrix (Iter %d)', iter);
    cm.DiagonalColor = [0 0.6 0];
    fig_Position = fig.Position; fig_Position(3) = fig_Position(3)*1.5; fig.Position = fig_Position;
    acc_string = sprintf('Accuracy=%.3f', accuracy);
    annotation('textbox', [0.78, 0.12, 0.9, 0.05], 'String', acc_string, 'Color', [0 0 0], 'FontWeight', 'bold', 'EdgeColor', 'none', 'FontSize', 12);
    nombre='GA_confusion_'; myfile = strcat(path_resultados, nombre);
    Name = strcat(myfile, string(iter)); Name1 = strcat(Name, '.jpg');
    saveas(fig, Name1); close(fig);

end


fprintf(fid, '\n\n BLOQUE 1: average of %d iterations ===\n', N1);
fprintf(fid, 'Accuracy: %.2f%% | Precision: %.2f%% | Recall: %.2f%% | F1: %.2f%%\n\n', ...
    mean(accuracy_iter)*100, mean(precision_iter)*100, mean(recall_iter)*100, mean(F1_scores_iter)*100);

fprintf(fid, 'TRAINING (average classes*iterations): Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%%\n\n', ...
    mean(accuracy_training)*100, mean(precision_training)*100, mean(recall_training)*100, mean(f1_training)*100);



% Block 2: Validation 
indicesTop = find(selection_frequency > 0);
num_features_top = numel(indicesTop);

fprintf(fid, '\n\n BLOCK 2: Validation %d features\n', num_features_top);

accuracy_validation = [];
precision_validation = [];
recall_validation = [];
f1_validation = [];

N2 = 150;
for iter = 1:N2
    
    indices = randperm(num_data);
    data_train    = data(indices(1:max_index_data(1)), :);
    data_test     = data(indices((max_index_data(1)+1):max_index_data(2)), :);
    data_validate = data(indices((max_index_data(2)+1):end), :);

    data_eval = [data_test; data_validate];

    if isempty(indicesTop)
        warning('No features in indicesTop (iter=%d).', iter); break;
    end

    [Ypred_eval, TestingAccuracy] = eval_ELM_once(data_train, data_eval, indicesTop, elmParams);

    Ytrue_eval = data_eval(:, end);
    accuracy = TestingAccuracy;

    % Metrics (macro)
    clases = unique(Ytrue_eval);
    numClases = numel(clases);

    accuracy_iteration = [];
    precision_iteration = [];
    recall_iteration = [];
    f1_iteration = [];

    TP_iter=0; TN_iter=0; FP_iter=0; FN_iter=0;

    for iC = 1:numClases
        clase = clases(iC);
        TP = sum((Ypred_eval == clase) & (Ytrue_eval == clase));
        TN = sum((Ypred_eval ~= clase) & (Ytrue_eval ~= clase));
        FP = sum((Ypred_eval == clase) & (Ytrue_eval ~= clase));
        FN = sum((Ypred_eval ~= clase) & (Ytrue_eval == clase));

        TP_iter=TP_iter+TP; TN_iter=TN_iter+TN; FP_iter=FP_iter+FP; FN_iter=FN_iter+FN;

        acc_c = (TP+TN) / (TP+TN+FP+FN + eps);
        prec  = TP / (TP + FP + eps);
        rec   = TP / (TP + FN + eps);
        F1    = 2*(prec*rec)/(prec+rec + eps);

        accuracy_iteration = [accuracy_iteration acc_c];
        precision_iteration = [precision_iteration prec];
        recall_iteration = [recall_iteration rec];
        f1_iteration = [f1_iteration F1];
    end

    accuracy_validation(iter) = mean(accuracy_iteration);
    precision_validation(iter)= mean(precision_iteration);
    recall_validation(iter)   = mean(recall_iteration);
    f1_validation(iter)       = mean(f1_iteration);

    fprintf(fid, 'VALIDATION iter=%d | Accuracy=%.2f%%\n', iter, accuracy*100);

    
end

fprintf(fid, '\n=== BLOCK 2: Average (iterations) ===\n');
fprintf(fid, 'VALIDATION Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%%\n\n', ...
    mean(accuracy_validation)*100, mean(precision_validation)*100, mean(recall_validation)*100, mean(f1_validation)*100);

fprintf(fid, 'The %d features are:\n', num_features_top);
for i = 1:num_features_top
    fprintf(fid, 'Frecuency #%d (%.0f Hz)\n', (gab_ind_frec + indicesTop(i)), round(frecuencias(gab_ind_frec + indicesTop(i))));
end


ancho = 2100; alto = 300;
fig = figure('Position', [100, 100, ancho, alto], 'Visible','off');
hold on;
x = 1:1:num_charact;
y = ones(1, num_charact)*0.02;
bar(x, y);    
y = zeros(1, num_charact);
y(indicesTop) = 1;
b2 = bar(x, y); b2.FaceColor = 'red';
hold off;

gap_x_axis = 12;
xticks(1:gap_x_axis:num_charact);
idx = (gab_ind_frec+1):gap_x_axis:(gab_ind_frec+num_charact);
xticklabels(compose('%.1f', frecuencias(idx)/1000));
xlabel('Frequency (kHz)', 'FontWeight', 'bold', 'FontSize', 18);
ylabel('Weight', 'FontWeight', 'bold', 'FontSize', 18);
ylim([0 1.01]); yticks(0:0.5:1);
title('Significance spectral component (GA+ELM)', 'FontSize', 20);
ax = gca; ax.XAxis.FontSize = 16; ax.YAxis.FontSize = 16;
nombre='GA_significance_'; myfile=strcat(path_resultados,nombre);
Name=strcat(myfile,string(N2)); Name1=strcat(Name,'.jpg');
saveas(fig, Name1); close(fig);


fclose(fid);







function subsets = build_subsets(m, lengths)
    subsets = {};
    for L = lengths
        for startIdx = 1:(m - L + 1)
            subsets{end+1} = startIdx:(startIdx + L - 1); 
        end
    end
end



function folds = build_stratified_kfolds(y, k)
    clases = unique(y);
    idx_folds = zeros(size(y));
    for iC = 1:numel(clases)
        idx = find(y == clases(iC));
        idx = idx(randperm(numel(idx)));
        % k blocks
        edges = round(linspace(0, numel(idx), k+1));
        for f = 1:k
            if edges(f)+1 <= edges(f+1)
                idx_folds(idx(edges(f)+1:edges(f+1))) = f;
            end
        end
    end
    folds.k = k;
    folds.idx = cell(k,1);
    folds.train = cell(k,1);
    folds.val = cell(k,1);
    N = numel(y);
    allIdx = (1:N)';
    for f = 1:k
        valIdx = find(idx_folds == f);
        trainIdx = setdiff(allIdx, valIdx);
        folds.idx{f} = valIdx;
        folds.train{f} = trainIdx;
        folds.val{f} = valIdx;
    end
end

function [bestMask, bestGroup, bestAccCV, bestSize] = run_ga_feature_selection( ...
    data_train, subsets, gaParams, folds, time_limit_seconds, deadline)

    popSize = gaParams.popSize;
    maxGens = gaParams.maxGens;
    tourK   = gaParams.tournament;
    Pc      = gaParams.Pc;
    Pm      = gaParams.Pm;
    G       = gaParams.G;
    minFeat = gaParams.minFeat;
    beta    = gaParams.beta;
    alpha   = gaParams.alpha;
    kfold   = gaParams.kfold;
    m       = gaParams.m;
    elm     = gaParams.elm;

    num_subsets = numel(subsets);

    % sparsidad
    p_active = 0.10;
    POP = randi([0 G], popSize, num_subsets);
    mask_zero = rand(popSize, num_subsets) > p_active;
    POP(mask_zero) = 0;

    % Initial evaluation 
    fitness = -inf(popSize,1);
    accBestGroup = zeros(popSize,1);
    sizeBestGroup = zeros(popSize,1);
    maskBestGroup = false(popSize, m);

    for i = 1:popSize
        [fitness(i), accBestGroup(i), sizeBestGroup(i), maskBestGroup(i,:)] = ...
            eval_fitness(POP(i,:), subsets, G, minFeat, alpha, beta, m, data_train, folds, elm);
    end

    % Elitism
    [bestFit, idxBest] = max(fitness);
    bestMask = maskBestGroup(idxBest,:);
    bestGroup = NaN; 
    bestAccCV = accBestGroup(idxBest);
    bestSize  = sizeBestGroup(idxBest);

    gen = 1;
    while gen <= maxGens
        if toc(deadline) > time_limit_seconds
            
            break;
        end

        % New population (elitism of 2)
        [~, ord] = sort(fitness, 'descend');
        newPOP = zeros(size(POP));
        newPOP(1,:) = POP(ord(1),:);
        newPOP(2,:) = POP(ord(2),:);

        
        for i = 3:2:popSize
            p1 = tournament_select(POP, fitness, tourK);
            p2 = tournament_select(POP, fitness, tourK);

            c1 = p1; c2 = p2;
            if rand < Pc
                [c1, c2] = uniform_crossover(p1, p2);
            end
            c1 = mutate_individual(c1, G, Pm);
            c2 = mutate_individual(c2, G, Pm);

            newPOP(i,:) = c1;
            if i+1 <= popSize
                newPOP(i+1,:) = c2;
            end
        end

        POP = newPOP;

        % Evaluation
        for i = 1:popSize
            [fitness(i), accBestGroup(i), sizeBestGroup(i), maskBestGroup(i,:)] = ...
                eval_fitness(POP(i,:), subsets, G, minFeat, alpha, beta, m, data_train, folds, elm);

            % Update
            if fitness(i) > bestFit
                bestFit = fitness(i);
                bestMask = maskBestGroup(i,:);
                bestAccCV = accBestGroup(i);
                bestSize  = sizeBestGroup(i);
            end
        end

        gen = gen + 1;
    end
end





function [fit, accBest, sizeBest, maskBest] = eval_fitness(chrom, subsets, G, minFeat, alpha, beta, m, data_train, folds, elm)

    accBest = -inf; sizeBest = 0; maskBest = false(1,m);
    anyValid = false;

    for g = 1:G
        idx_subs = find(chrom == g);
        if isempty(idx_subs), continue; end
        mask = false(1,m);
        for k = 1:numel(idx_subs)
            mask(subsets{idx_subs(k)}) = true;
        end
        S = find(mask);
        if numel(S) < minFeat
            continue;
        end

        % Accuracy internal k-fold
        accCV = eval_CV_accuracy_ELM(data_train, S, folds, elm);

        
        fit_g = alpha*accCV - beta*(numel(S)/m);

        if accCV > accBest
            accBest = accCV;
            sizeBest = numel(S);
            maskBest = mask;
        end

        anyValid = true; 
    end

    if ~anyValid
        fit = -inf;
    else
        fit = alpha*accBest - beta*(sizeBest/m);
    end
end



function accCV = eval_CV_accuracy_ELM(data_train, feat_idx, folds, elm)
% k-fold on TRAIN con ELM
    k = folds.k;
    acc = zeros(k,1);
    for f = 1:k
        tr = folds.train{f};
        va = folds.val{f};

        train_fold = [data_train(tr, feat_idx)  data_train(tr,end)];
        test_fold  = [data_train(va, feat_idx)  data_train(va,end)];

        % Re-order 
        train_fold = [train_fold(:, end) train_fold(:, 1:end-1)];
        test_fold  = [test_fold(:,  end) test_fold(:,  1:end-1)];

        [~, TestingAccuracy] = ELM(train_fold, test_fold, elm.Elm_Type, elm.NumberofHiddenNeurons, elm.ActivationFunction);
        acc(f) = TestingAccuracy;
    end
    accCV = mean(acc);
end





function [Ypred, TestingAccuracy] = eval_ELM_once(data_train, data_eval, feat_idx, elm)

    tr = [data_train(:, feat_idx) data_train(:, end)];
    te = [data_eval(:,  feat_idx) data_eval(:,  end)];

    tr = [tr(:, end) tr(:, 1:end-1)];
    te = [te(:, end) te(:, 1:end-1)];

    [TY_label_index, TestingAccuracy] = ELM(tr, te, elm.Elm_Type, elm.NumberofHiddenNeurons, elm.ActivationFunction);
    Ypred = TY_label_index(:);
end

function parent = tournament_select(POP, fitness, k)
    n = size(POP,1);
    cand = randi(n, [k 1]);
    [~, idx] = max(fitness(cand));
    parent = POP(cand(idx), :);
end

function [c1, c2] = uniform_crossover(p1, p2)
    mask = rand(size(p1)) > 0.5;
    c1 = p1; c2 = p2;
    c1(mask) = p2(mask);
    c2(mask) = p1(mask);
end

function ind = mutate_individual(ind, G, Pm)
    mut_mask = rand(size(ind)) < Pm;
    if any(mut_mask)
        ind(mut_mask) = randi([0 G], [1, sum(mut_mask)]);
    end

end
