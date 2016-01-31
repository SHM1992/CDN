function [Res1,Res2,Res3] = clustering(x,y,boxes,k1,k2,k3)
%this function is used to cluster the input training samples into several classes 
%for training different JDNs
%the clustering criterion are the 3 input channels defined by JDN

%the first channel
%k-means:
num1 = size(x{1},3);
if num1 < k1
    fprintf('error');
    return;
end

%initial value of centroid
p1 = randperm(num1);
for i = 1:k1
    c1(:,:,i) = x{1}(:,:,p1(i));
end

temp = zeros(num1,1);%temp is used to be category vector for each iteration

%clustering
while 1
    d1 = DistMatrix(x{1},c1);
    min_d1 = min(d1,[],2);
    class1 = zeros(num1,1);
    for i = 1:num1
        class1(i) = find(d1(i,:) == min_d1(i),1);
    end
    if class1 == temp
        break;
    else
        temp = class1;
    end
    for i = 1:k1
        if isempty(find(class1 == i))
            continue;
        else
            c1(:,:,i) = mean(x{1}(:,:,find(class1 == i)),3);
        end
    end
end

n = 1;
for p = 1:k1
    inds = find(class1 == p);
    if isempty(inds)
        continue;
    end
    for q = 1:length(inds)
        Res1{p}.x{1}(:,:,n) = x{1}(:,:,inds(q));
        Res1{p}.x{2}(:,:,n) = x{2}(:,:,inds(q));
        Res1{p}.x{3}(:,:,n) = x{3}(:,:,inds(q));
        Res1{p}.y(:,n) = y(:,inds(q));
        Res1{p}.boxes(n,:) = boxes(inds(q),:);
        Res1{p}.num = length(inds);
        n = n + 1;
    end
end

%the second channel
%k-means:
num2 = size(x{2},3);
if num2 < k2
    fprintf('error');
    return;
end

%initial value of centroid
p2 = randperm(num2);
for i = 1:k2
    c2(:,:,i) = x{2}(:,:,p2(i));
end

temp = zeros(num2,1);%temp is used to be category vector for each iteration

%clustering
while 1
    d2 = DistMatrix(x{2},c2);
    min_d2 = min(d2,[],2);
    class2 = zeros(num2,1);
    for i = 1:num2
        class2(i) = find(d2(i,:) == min_d2(i),1);
    end
    if class2 == temp
        break;
    else
        temp = class2;
    end
    for i = 1:k2
        if isempty(find(class2 == i))
            continue;
        else
            c2(:,:,i) = mean(x{2}(:,:,find(class2 == i)),3);
        end
    end
end

n = 1;
for p = 1:k2
    inds = find(class2 == p);
    if isempty(inds)
        continue;
    end
    for q = 1:length(inds)
        Res2{p}.x{1}(:,:,n) = x{1}(:,:,inds(q));
        Res2{p}.x{2}(:,:,n) = x{2}(:,:,inds(q));
        Res2{p}.x{3}(:,:,n) = x{3}(:,:,inds(q));
        Res2{p}.y(:,n) = y(:,inds(q));
        Res2{p}.boxes(n,:) = boxes(inds(q),:);
        Res2{p}.num = length(inds);
        n = n + 1;
    end
end

%the third channel
%k-means:
num3 = size(x{3},3);
if num3 < k3
    fprintf('error');
    return;
end

%initial value of centroid
p3 = randperm(num3);
for i = 1:k3
    c3(:,:,i) = x{3}(:,:,p3(i));
end

temp = zeros(num3,1);%temp is used to be category vector for each iteration

%clustering
while 1
    d3 = DistMatrix(x{3},c3);
    min_d3 = min(d3,[],2);
    class3 = zeros(num3,1);
    for i = 1:num3
        class3(i) = find(d3(i,:) == min_d3(i),1);
    end
    if class3 == temp
        break;
    else
        temp = class3;
    end
    for i = 1:k3
        if isempty(find(class3 == i))
            continue;
        else
            c3(:,:,i) = mean(x{3}(:,:,find(class3 == i)),3);
        end
    end
end

n = 1;
for p = 1:k3
    inds = find(class3 == p);
    if isempty(inds)
        continue;
    end
    for q = 1:length(inds)
        Res3{p}.x{1}(:,:,n) = x{1}(:,:,inds(q));
        Res3{p}.x{2}(:,:,n) = x{2}(:,:,inds(q));
        Res3{p}.x{3}(:,:,n) = x{3}(:,:,inds(q));
        Res3{p}.y(:,n) = y(:,inds(q));
        Res3{p}.boxes(n,:) = boxes(inds(q),:);
        Res3{p}.num = length(inds);
        n = n + 1;
    end
end


function d = DistMatrix(x,c)
%this function is used to compute the distance between data sample matrix
%and centroid matrix

for i = 1:size(x,3)
    for j = 1:size(c,3)
        d(i,j) = norm(x(:,:,i)-c(:,:,j),2);
    end
end
