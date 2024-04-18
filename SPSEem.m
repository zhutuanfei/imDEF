function [competentLevel] = SPSEem(TrainData,CL,Label,TestData,classes,Learner,alpha2,b,beta)
[n,T] = size(CL);
% T = spe_n*bagging_n;
% preL = zeros(size(TestData,1),T);
% beta = 2; 
competentLevel = zeros(size(TestData,1),T);
% temp_Label = [];
% for k = 1:bagging_n  % 加宽，以bagging为单位加�?
%     temp_Label = [temp_Label repmat(CL(:,(k-1)*spe_n+1:k*spe_n),1,beta)];
% end
% CL  = temp_Label;

a = 0.8;
%-------构�?�k_p个类似bagging的分类器
% for k=1:bagging_n
%     cur_ind = (k-1)*spe_n*beta+1:k*spe_n*beta;
    cur_ind = 1:T;
    o_ind = (sum(CL(:,cur_ind))./n)>=0.99;
    competentLevel(:,cur_ind(o_ind)) = 1;
    o_ind = (sum(CL(:,cur_ind))./n)<=0.01;
    competentLevel(:,cur_ind(o_ind)) = 0;
    o_ind = ((sum(CL(:,cur_ind))./n)<0.99) & ((sum(CL(:,cur_ind))./n)>0.01);
    if sum(o_ind)~=0
        tr_ind = cur_ind(o_ind);
%         rand_k = randperm(numel(tr_ind));
       [competentLevel(:,tr_ind)]  = SPEv42(TrainData,CL(:,tr_ind),Label,TestData,classes,beta,Learner,a);
    end

%     disp (['TrainDataSPE #' int2str(1) ':' int2str(bagging_n*spe_n*beta)]);
% end
end

function [competentLevel] = SPEv42(TrainData,CL,Label,TestData,classes,beta,Learner,a)
class_n = numel(classes);
T = size(CL,2);
model = cell(beta,T);
modelLabel = cell(beta,T);
% preTrL = zeros(size(CL,1),T);   %prediction on validation data
% preTrM = zeros(size(CL,1),T*class_n);
temp_label = CL;
temp_label(temp_label==1)=-1;
temp_label(temp_label==0)=1;
% hardness = sum(predict_pro.*temp_label,2);
hardness = [];
hardness(:,1) = sum(temp_label,2);

bin_n = 20;
% a = (b*beta);
% a = 2;
powers = 1-(0:beta-1)./(a*(beta-1)); 
oc = unique(Label);
w = zeros(1,numel(oc));
for i = 1:numel(oc)
    w(i) = sum(Label==oc(i));
end
w = max(sqrt(w))./sqrt(w);
sw = zeros(numel(Label),1);
for i = 1:numel(Label)
    sw(i) = w(oc==Label(i));
end

% hardness(:,1) = inithardness;
% BB = cell(1,T);
% PL = cell(1,T);
% thre = (1/bin_n)*(0:bin_n);
tr_n = ceil(size(TrainData,1));
for r = 1:beta
%     H = sum(hardness(:,end),2);
%     H = hardness(:,end);
    H = mean(hardness,2);
%     disp(['iteration-' num2str(t) ' :average(H): ' num2str(mean(H))])
%     if min(H)<0
%        H = H - min(H);
%        H = H./max(H);
%     end
    H = (H-min(H))./(max(H)-min(H));
    [B,AHard] = cutToBins(H,bin_n);
    
%     sp_f = tan(t*pi/(2*T));
    pl = (1./(1e-2+AHard)).^powers(r);
    pl = pl./nansum(pl);
%     BB{t} = B;
%     PL{t} = pl;
    predict_pro = [];
    for t = 1:T
    label_t = CL(:,t);
    
    while 1
       count = 0;
       re_ind = [];
%     for j = 1:class_n
       for i = 1:bin_n
           if ~isnan(pl(i))
               if numel(B{i})==1
                  ind_i = repmat(B{i},ceil(pl(i)*tr_n),1);
               else
                  ind_i = randsample(B{i},ceil(pl(i)*tr_n),true,sw(B{i}));
               end
              re_ind = [re_ind; ind_i];
           end
        end
%     end
       if numel(unique(label_t(re_ind)))==class_n
           break;
       else
           count = count + 1;
       end
       if count>100
           disp('error!')
       end
    end
    [~,~,model{r,t},modelLabel{r,t},~,preTrM_t]=runLebetaer_RM(Learner,TrainData(re_ind,:),label_t(re_ind),2,TrainData);  
    mapind = modelLabel2classes(modelLabel{r,t},classes); preTrM_t(:,mapind) = preTrM_t;
    predict_pro(:,t) = preTrM_t(:,classes==1);
    end
    hardness(:,r+1) = sum(predict_pro.*temp_label,2);
    disp([int2str(r) '-th round!'])
end
for r = 1:beta+1
   disp ([int2str(r) '-th round, avg-hardness: ' int2str(mean(hardness(:,r))) ', min-hardness: ' int2str(max(hardness(:,r)))]);  
end
competentLevel = zeros(size(TestData,1),T);
% competentLevel = preL;
for r = 1:beta
for t = 1:T
    [~,preTestM] = LearnerPredict(Learner,model{r,t},TestData,ones(size(TestData,1),1),modelLabel{r,t});
%     preL(:,t) = preTestL;
    competentLevel(:,t) = competentLevel(:,t) + preTestM(:,modelLabel{r,t}==1);
end
end
competentLevel = competentLevel./beta;
end




function [B,AHard] = cutToBins(H,bin_n)
% H is the computed hardness by F
B = cell(bin_n,1);
AHard = zeros(bin_n,1);
% for j = 1:numel(classes)
%     ind_j = CL==classes(j);
%     max_hard = max(H); min_hard = min(H);
%     thre =  (0:bin_n)./bin_n;
    thre = linspace(min(H),max(H),bin_n+1);
    thre(end) = thre(end)+0.01;
    for i = 1:bin_n
        cur_ind = find(H<thre(i+1) & H >= thre(i));
%         if ~isempty(cur_ind)
        B{i} = cur_ind; 
        AHard(i) = sum(H(cur_ind))/numel(cur_ind);
%         end
    end
% end
end