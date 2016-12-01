%doc tat ca anh va luu vao ma tran J(10000,1,28,28) trong file .mat

srcFiles = dir('./test/*.jpg');
name = {srcFiles.name}';
name = sort_nat(name);

I = zeros(length(srcFiles),1,28,28);
for i = 1 : length(srcFiles)
    filename = strcat('./test/',name{i});
    I(i,1,:,:) = imread(filename);
end
save('test_img.mat','I');