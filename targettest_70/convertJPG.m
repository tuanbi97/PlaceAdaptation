folders = {'barn', 'beach', 'bedroom', 'castle', 'classroom', 'desert', 'kitchen', 'library', 'mountain', 'river'}
for j = 1:length(folders)
	st = ['./' folders{j} '/*.png']
	s = dir(st);
	for i = 1:length(s)
		imname = s(i).name;
		path = ['./' folders{j} '/' imname];
		disp(path)
		im = imread(path);
		imwrite(im, [path(1:length(path)-4) '.jpg']);
	end
end
