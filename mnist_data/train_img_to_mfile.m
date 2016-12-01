%doc tat ca anh va luu vao ma tran J(60000,1,28,28) trong file .mat

srcFiles = dir('./train/*.jpg');
name = {srcFiles.name}';
name = sort_nat(name);

J = zeros(length(srcFiles),1,28,28);
for i = 1 : length(srcFiles)
    filename = strcat('./train/',name{i});
    J(i,1,:,:) = imread(filename);
end
save('train_img.mat','J');