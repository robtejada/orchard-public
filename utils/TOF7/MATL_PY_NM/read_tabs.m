%% Read Nadine's tab_*.dat files and save in usable structs
clear
clc

%% The f_n and f'_n coefficients
% After the header and global 7, each block is headed by a 3-int row. The first
% is the n in f_n (0 to 14); the second is always zero; the third is the number
% of rows in the block. A row consists of an unused number followed by 7 powers
% to raise s_i by followed by the coefficient to multiply those products by to
% get f_n or f'_n.

f_coeffs = struct();

%% f_n first
lines = readlines('tab_fn.dat','WhitespaceRule','trim','EmptyLineRule','skip');
cl = 1;
while lines(cl).startsWith('#') % skip header
    cl = cl + 1;
end

% Read and parse 8 blocks (f_0:f_14)
for block = 1:8
    cl = cl + 1;
    blockhead = lines(cl).split();
    fn = sprintf('f%s',blockhead(1)); % char fn for field name
    bl = double(blockhead(3)); % nlines in block
    f_coeffs.(fn) = nan(bl,8);
    for k=1:bl
        cl = cl + 1;
        line = lines(cl).split();
        f_coeffs.(fn)(k,:) = double(line(2:end))';
    end
end

%% Next f'_n
lines = readlines('tab_fnp.dat','WhitespaceRule','trim','EmptyLineRule','skip');
cl = 1;
while lines(cl).startsWith('#') % skip header
    cl = cl + 1;
end

% Read and parse 8 blocks (f'_0:f'_14)
for block = 1:8
    cl = cl + 1;
    blockhead = lines(cl).split();
    fn = sprintf('f%sp',blockhead(1)); % char fn for field name
    bl = double(blockhead(3)); % nlines in block
    f_coeffs.(fn) = nan(bl,8);
    for k=1:bl
        cl = cl + 1;
        line = lines(cl).split();
        f_coeffs.(fn)(k,:) = double(line(2:end))';
    end
end

%% The A_k coefficients (equiv. of B12-B15 in Nettellmann 2017)
% After the header and global 7, each block is headed by a 3-int row. The first
% is the n in S_n or S'_n; the second is k in A_k; the third is the number of
% rows in the block. A row consists of an unused number followed by 7 powers to
% raise s_i by followed by the coefficient to multiply those products by, to
% get one term in the big parentheses in front of S_n or S'_n, in the RHS of
% the non-linear equations to solve (for s_i).

A_coeffs = struct();

%% S_n first
lines = readlines('tab_Sn.dat','WhitespaceRule','trim','EmptyLineRule','skip');
cl = 1;
while lines(cl).startsWith('#') % skip header
    cl = cl + 1;
end

% Read and parse all blocks
while cl < length(lines)
    cl = cl + 1;
    blockhead = lines(cl).split();
    Sn = sprintf('S%s',blockhead(1)); % char Sn for field name
    Ak = sprintf('A%s',blockhead(2)); % char Ak for field name
    bl = double(blockhead(3)); % nlines in block
    A_coeffs.(Ak).(Sn) = nan(bl,8);
    for k=1:bl
        cl = cl + 1;
        line = lines(cl).split();
        A_coeffs.(Ak).(Sn)(k,:) = double(line(2:end))';
    end
end

%% S'_n next
lines = readlines('tab_Snp.dat','WhitespaceRule','trim','EmptyLineRule','skip');
cl = 1;
while lines(cl).startsWith('#') % skip header
    cl = cl + 1;
end

% Read and parse all blocks
while cl < length(lines)
    cl = cl + 1;
    blockhead = lines(cl).split();
    Snp = sprintf('S%sp',blockhead(1)); % char Snp for field name
    Ak = sprintf('A%s',blockhead(2)); % char Ak for field name
    bl = double(blockhead(3)); % nlines in block
    A_coeffs.(Ak).(Snp) = nan(bl,8);
    for k=1:bl
        cl = cl + 1;
        line = lines(cl).split();
        A_coeffs.(Ak).(Snp)(k,:) = double(line(2:end))';
    end
end

%% and finally, "m"-term
lines = readlines('tab_m.dat','WhitespaceRule','trim','EmptyLineRule','skip');
cl = 1;
while lines(cl).startsWith('#') % skip header
    cl = cl + 1;
end

% Read and parse all blocks
while cl < length(lines)
    cl = cl + 1;
    blockhead = lines(cl).split();
    Ak = sprintf('A%s',blockhead(2)); % char Ak for field name
    bl = double(blockhead(3)); % nlines in block
    A_coeffs.(Ak).m = nan(bl,8);
    for k=1:bl
        cl = cl + 1;
        line = lines(cl).split();
        A_coeffs.(Ak).m(k,:) = double(line(2:end))';
    end
end
