function dc7 = pythonize_c7(C7,dumpname)
% Convert a C7 struct to python dict with real lists instead of memoryviews.
%
% When Matlab convertes a struct to a python dict 2d arrays are converted to a
% python memoryview object. Their data is not copied. This is usually fine but
% it makes it hard to serialize with pickle or json. So here we loop over the
% relevant dict keys and manually populate them with the full data, by calls to
% the memoryview.tolist() method, making them nested lists.
%
% With nonempty dumpname the output dict will also be json.dumped to that file.

if nargin < 2, dumpname = ''; end

dc7 = py.dict(C7);

fnames = fieldnames(C7.f);
anames = fieldnames(C7.A);
snames = fieldnames(C7.A.A0);

for k=1:length(fnames)
    dc7{'f'}{fnames{k}} = dc7{'f'}{fnames{k}}.tolist();
end
for j=1:length(anames)
    for k=1:length(snames)
        dc7{'A'}{anames{j}}{snames{k}} = dc7{'A'}{anames{j}}{snames{k}}.tolist();
    end
end

if ~isempty(dumpname) % json it
    f = py.open(dumpname,'w');
    try
        py.json.dump(dc7,f);
        f.close();
    catch
        f.close()
    end
end

end
