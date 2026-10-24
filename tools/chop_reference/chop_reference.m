function chop_reference(inputs, outputs)
% One CHOP step with the author's MATLAB code, for APEX's reference test.
%
% The model update is RLM_MAC_CHOP.m's (5SpotExample/run_scripts, lines 177-271: the
% per-member taper on the shared RLM-MAC gain, with getCorrMtx and rationalFunction); the
% hyper-parameter updates are update_hyper.m and update_hyper_no_loc.m, called as they are.
% Run by tools/chop_reference/generate.py, with the author's run_scripts on the path.
load(inputs, 'ensemble', 'simData', 'perturbedData', 'simDataNew', 'lambda', 'tsvdData', 'nm');
kalmanOptions.tsvdData = tsvdData;
kalmanOptions.autoAdaLoc = 1;
ne = size(ensemble, 2) - 1;
nd = size(simData, 1);
model_index = 1:nm;
hyper_index = nm + 1 : size(ensemble, 1);

% --- RLM_MAC_CHOP.m, lines 177-271 (the model update of one trial) ---
deltaM = zeros(size(ensemble, 1), ne);
meanEn = mean(ensemble, 2);
for j = 1:ne
    deltaM(:, j) = ensemble(:, j) - meanEn;
end
simMean = simData(:, end);
deltaD = zeros(nd, ne);
for j = 1:ne
    deltaD(:, j) = simData(:, j) - simMean;
end
corr_mtx = getCorrMtx(ensemble(model_index, 1:ne), simData(:, 1:ne));
[Ud, Wd, Vd] = svd(deltaD, 'econ');
val = diag(Wd);
total = sum(val);
for j = 1:ne
    svdPd = j;
    if (sum(val(1:j)) / total > kalmanOptions.tsvdData)
        break
    end
end
Vd = Vd(:, 1:svdPd);
Ud = Ud(:, 1:svdPd);
Wd = val(1:svdPd);
alpha = lambda * sum(Wd.^2) / svdPd;
alphaRaw = alpha;
alpha = max(alpha, 1e-3);
x1 = zeros(ne, svdPd);
for jVec = 1:svdPd
    for j = 1:ne
        x1(j, jVec) = Vd(j, jVec) * Wd(jVec) / (alpha + Wd(jVec)^2);
    end
end
X = perturbedData - simData(:, 1:ne);
ens_bg = ensemble(model_index, :);
ens_hyper = ensemble(hyper_index, :);
coeff = deltaM(model_index, :) * x1 * Ud';
model_change = zeros(size(corr_mtx, 1), ne);
factor = 2;
for i = 1:ne
    lc = squeeze(abs((ens_hyper(:, i)' + 1e-6) * factor));
    tapering_mtx = rationalFunction(1 - abs(corr_mtx), lc);
    tapering_mtx = reshape(tapering_mtx, size(corr_mtx));
    model_change(:, i) = (tapering_mtx .* coeff) * X(:, i);
end
ens_bg(:, 1:ne) = ens_bg(:, 1:ne) + model_change;
modelUpdated = ens_bg(:, 1:ne);
modelCorr = corr_mtx;
modelRank = svdPd;

% --- the hyper-parameter updates, from the new forecast simDataNew ---
hyperLocalized = update_hyper(ens_hyper, simDataNew, perturbedData, lambda, kalmanOptions);
hyperUnlocalized = update_hyper_no_loc(ens_hyper, simDataNew, perturbedData, lambda, kalmanOptions);
hyperLocalized = hyperLocalized(:, 1:ne);
hyperUnlocalized = hyperUnlocalized(:, 1:ne);
hyperCorr = getCorrMtx(ens_hyper(:, 1:ne), simDataNew(:, 1:ne));

save(outputs, 'modelUpdated', 'modelCorr', 'modelRank', 'alphaRaw', 'hyperLocalized', ...
     'hyperUnlocalized', 'hyperCorr', '-v7');
end
