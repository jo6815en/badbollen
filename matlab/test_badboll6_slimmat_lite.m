%% Init multipol
% 
format compact
addpath('multipol');
addpath('tools');


%% Generate data so that there is one known ground truth vector

% Choose 6 random directions. These are the intersections of three circles
dd = randn(3,6);
dd = psphere(dd); %Normalize so that each column has length one

% Generate for each dd(:,j) three orthogonal directions. 
% These are the three normals to three circles for each line.
data = zeros(3,6,3);
for i = 1:3;
    for j = 1:6;
        tmp = cross(randn(3,1),dd(:,j)); % Cross product makes tmp orthogonal to dd(:,j)
        tmp = tmp/norm(tmp);
        data(:,j,i)=tmp;
    end
end
% Now data(:,j,i) are the normals for circle j in image i. 
% The three images have the correct orientation.
% i.e. the ground truth would be gt_sol = [1 0 0 0 1 0 0 0];

% Generate two random rotations and rotate view 2 and 3.
tmp1 = randn(4,1); tmp1 = tmp1/norm(tmp1); % Random unit quaternion
tmp2 = randn(4,1); tmp2 = tmp2/norm(tmp2); % Random unit quaternion
r1 = quat2rot(tmp1); % rotation matrix from view 2 to 1
r2 = quat2rot(tmp2); % rotation matrix from view 3 to 1
data(:,:,2)= r1'*data(:,:,2); % Multiply with the inverse of r1
data(:,:,3)= r2'*data(:,:,3); % Multiply with the inverse of r2

% We now have the generated data
nn = [data(:,:,1) data(:,:,2) data(:,:,3)]';
% with one ground truth
gt_sol = [tmp1;tmp2];


%% This is old code to generate random data and code to export to txt 
% so that it can be run by python version
%nn = normc(randn(18,3));
%fname = 'config_seed1.txt';
%write_config(nn,fname);
%[q2, q3, points] = read_solutions("am_sols_config_seed1_real.txt");
% for i = 1:size(q2,1);
%     asol = [q2(i,:) q3(i,:)]';
%     evaluate(eqs,asol)
% end

%%

t0 = tic;

%% Generate the six equations

n = 8;
x = create_vars(n);
Q1 = [x(1:4)];
Q2 = [x(5:8)];
R1 = quat2rot(Q1);
R2 = quat2rot(Q2);

u = nn(1:6,:)';
v = nn(7:12,:)';
w = nn(13:18,:)';
vp = R1*v;
wp = R2*w;

eqs = [det([u(:,1) vp(:,1) wp(:,1)]); ...
       det([u(:,2) vp(:,2) wp(:,2)]); ...
       det([u(:,3) vp(:,3) wp(:,3)]); ...
       det([u(:,4) vp(:,4) wp(:,4)]); ...
       det([u(:,5) vp(:,5) wp(:,5)]); ...
       det([u(:,6) vp(:,6) wp(:,6)])];

% ... and the two equations that need to be saturated.
s2 = Q1'*Q1;
s3 = Q2'*Q2;


%% Generate homogeneous monomials of degree n in four variables.
% 1: 4
% 2: 10
% 3: 20
% 4: 35
% 5: 56

p = create_upto_degree([1 1 1 1],1);
m1 = monomials(p);
tmp = find(sum(m1)==1);
m1 = m1(:,tmp);
p = create_upto_degree([2 2 2 2],2);
m2 = monomials(p);
tmp = find(sum(m2)==2);
m2 = m2(:,tmp);
p = create_upto_degree([3 3 3 3],3);
m3 = monomials(p);
tmp = find(sum(m3)==3);
m3 = m3(:,tmp);
p = create_upto_degree([4 4 4 4],4);
m4 = monomials(p);
tmp = find(sum(m4)==4);
m4 = m4(:,tmp);
p = create_upto_degree([5 5 5 5],5);
m5 = monomials(p);
tmp = find(sum(m5)==5);
m5 = m5(:,tmp);

%% Generate bi-homogeneous monomials in eight unknowns of 
% degree (3,0), (0,3), (4,0), (0,4), (5,0), (0,5)
m3c1 = x(1)*zeros(length(m3),1);
for i = 1:length(m3);
    m3c1(i)=multipol(1,[m3(:,i);zeros(4,1)]);
end
m3c2 = x(1)*zeros(length(m3),1);
for i = 1:length(m3);
    m3c2(i)=multipol(1,[zeros(4,1);m3(:,i)]);
end
m4c1 = x(1)*zeros(length(m4),1);
for i = 1:length(m4);
    m4c1(i)=multipol(1,[m4(:,i);zeros(4,1)]);
end
m4c2 = x(1)*zeros(length(m4),1);
for i = 1:length(m4);
    m4c2(i)=multipol(1,[zeros(4,1);m4(:,i)]);
end
m5c1 = x(1)*zeros(length(m5),1);
for i = 1:length(m5);
    m5c1(i)=multipol(1,[m5(:,i);zeros(4,1)]);
end
m5c2 = x(1)*zeros(length(m5),1);
for i = 1:length(m5);
    m5c2(i)=multipol(1,[zeros(4,1);m5(:,i)]);
end



%% Nu tar vi fram matrisen för 
% (I:s3) inom (5,4)
% För detta tar vi fram idealet I för bi-degree (5,6)
% Genom att ta alla monom av typ (3,4) och multiplicerar
% med ursprungsekvationerna som ju är av typ (2,2). 
%
% Sedan bildar vi alla monom av grad (5,4)
% och multiplicerar dessa med s3 som ju är av typ (0,2)
% Då får vi polynom av typ (5,6) som vi kollar om de är i I. 
%
% Det ger oss till slut ett antal ekvationer av grad (5,4) 
% som ligger i (I:s3). 

if 0,
% Alla monom av bi-grad (3,4)
Amultmon = m3c1*m4c2'; 
Amultmon = Amultmon(:);
% Multiplicera med ekvationerna. Det 
Aeqs = eqs*Amultmon';
Aeqs = Aeqs(:);
[A,amon] = polynomials2matrix(Aeqs);

% Man kan kolla att gt_sol uppfyller ekvationerna
% norm(evaluate(Aeqs,gt_sol))

% Kanske klurigt steg. Vi vill senare kolla om
% vilka polynom som ligger i (5,6) delen av I
% sådana polynom har coefficienter som är linjärkombinationer 
% av raderna i A
% Ett enkelt test är alltså om en sådan rad är vinkelrät mot
% ortogonala kopmlementet till A. Nu tar vi fram Z som är 
% en bas för ortogonala komplementet.
% A har storlek 	4200    4704
% A ska ha rang 	3600
% Ortogonala komplementet Z har då storlek 	4704	1104
% dvs det finns 1104 krav på en koefficientvektor som har längd 4704
% Kolla först rang
%
tmpsvda = log10(svd(A));
figure(1); clf; plot(tmpsvda,'*'); title('Singular values of A (5,4) case')
%figure(2); spy(A);
%
disp('Log10 of singular values 3600 and 3601 of A')
[tmpsvda(3600:3601)']
% Denna borde vara stor vid 3600 och liten vid 3601

% Nu tar vi fram ortogonala komplementet med qr
% Man skulle kunna köra svd eller något annat också. 
[Z, rankA, diagR, tol] = row_space_complement(A);

% Eventuella debugutskrifter
%fprintf('rank(A) estimate: %d\n', rankA);
%fprintf('size(Z): %d x %d\n', size(Z, 1), size(Z, 2));
%fprintf('relative A*Z residual: %.3e\n', ...
%    norm(A * Z, 'fro') / (norm(A, 'fro') * norm(Z, 'fro')));
%fprintf('orthogonality error: %.3e\n', ...
%    norm(Z' * Z - eye(size(Z, 2)), 'fro'));

% Nu genererar vi alla monom av typ (5,4)
% Now form Bm
Bmmultmon = m5c1*m4c2';
Bmmultmon = Bmmultmon(:);
% Spara också alla monom i en matris vmmon
vmmon = zeros(8,length(Bmmultmon));
for i = 1:length(Bmmultmon)
    vmmon(:,i)=monomials(Bmmultmon(i));
end

% Multiplicera alla monom med s3
[Bm Bmmon] = polynomials2matrix(Bmmultmon*s3);
% Bm har storlek 1960 x 4704
% en linjär kombination y av de 1960 monom ger en
% koefficientvektor y*Bm av storlek 4704
% Ett sådant polynom ligger i I om (y*Bm)*Z = 0
% då ska alltså kombinationerna y vara vänsternollvektorer till Bm*Z
C= Bm*Z;
% C har storlek 1960*1104
% Den har rang 600

% Här körde jag svd. Lite säkrare. Går kanske med QR också
[uu,ss,vv]=svd(C);
figure(3); plot(log10(diag(ss)),'*');
Y = uu(:,601:end);
disp('Log10 of singular values 600 and 601 of C')
log10([ss(600,600) ss(601,601)])
% Även här borde det första vara stort och det andra litet

% Då får vi till slut fram 1360 polynom av grad (5,4)
% som ligger i (I:s3)
vm = Y';
vmeqs = matrix2polynomials2(vm, vmmon);
% Man kan nu kolla att det inte blivit något strul 
% genom att se om gt_sol uppfyller dessa ekvationer.
% norm(evaluate(vmeqs,gt_sol))

% Nu kör vi qr med kolumnpivotering för att
% (i) hitta en bas av 600 monom av grad (5,4) för kvotrummet
% Jag flyttar om kolumnerna så att dessa hamnar längst till höger
[q,r,p] = qr(vm);
vm = q'*vm*p;
vmmon = vmmon*p;
% Man kan nu kolla att det inte blivit något strul 
% genom att se om gt_sol uppfyller dessa ekvationer.
% norm(evaluate(matrix2polynomials2(vm, vmmon),gt_sol))

% Storlekarna borde vara
% A: 	4200    4704
% rank 	3600
% Z: 	4704	1104
% Bm: 	1960	4704
% C: 	1960	1104
disp('Sizes of A, Z, Bm and C')
size(A)
size(Z)
size(Bm)
size(C)

end;
%% Sedan tar vi fram matrisen för 
% (I:s2) inom (5,5)
% Det är precis på samma sätt som ovan. Kommenterar inte så mycket nu

Amultmon = m5c1*m3c2';
Amultmon = Amultmon(:);
Aeqs = eqs*Amultmon';
Aeqs = Aeqs(:);
% norm(evaluate(Aeqs,gt_sol))

[A,amon] = polynomials2matrix(Aeqs);

tmpsvda = log10(svd(A));
figure(2); clf; plot(tmpsvda,'*'); title('Singular values of A (5,5) case')
%figure(2); spy(A);
disp('Log10 of singular values 5496 and 5497 of A')
[tmpsvda(5496:5497)']

[Z, rankA, diagR, tol] = row_space_complement(A);


%fprintf('rank(A) estimate: %d\n', rankA);
%fprintf('size(Z): %d x %d\n', size(Z, 1), size(Z, 2));
%fprintf('relative A*Z residual: %.3e\n', ...
%    norm(A * Z, 'fro') / (norm(A, 'fro') * norm(Z, 'fro')));
%fprintf('orthogonality error: %.3e\n', ...
%    norm(Z' * Z - eye(size(Z, 2)), 'fro'));

% Now form Bm

Bmmultmon = m5c1*m5c2';
Bmmultmon = Bmmultmon(:);

wmmon = zeros(8,length(Bmmultmon));
for i = 1:length(Bmmultmon)
    wmmon(:,i)=monomials(Bmmultmon(i));
end


[Bm Bmmon] = polynomials2matrix(Bmmultmon*s2);

C= Bm*Z;


[uu,ss,vv]=svd(C);
figure(3); plot(log10(diag(ss)),'*');
Y = uu(:,601:end);
disp('Log10 of singular values 600 and 601 of A')
log10([ss(600,600) ss(601,601)])

wm = Y';
[q,r,p] = qr(wm);
% norm(evaluate(matrix2polynomials2(wm, wmmon),gt_sol))

wm = q'*wm*p;
wmmon = wmmon*p;
% norm(evaluate(matrix2polynomials2(wm, wmmon),gt_sol))

% Storlekarna borde vara
% A: 	6720	6720
% rank 	5496
% Z: 	6720	1224
% Bm: 	3136    6720
% C: 	3136    1224
disp('Sizes of A, Z, Bm and C')
size(A)
size(Z)
size(Bm)
size(C)

%% Phuh. Nu har vi gjort det jobbiga. 
% Nu har vi det roliga kvar. 

% Vi har en bas av 600 monom för V (5,4) biten
%Bv = vmmon(:,(end-599):end);
load Bv
% och en bas för 600 monom för W (5,5) biten
Bw = wmmon(:,(end-599):end);

% Det finns 3136 monom av grad (5,5). 
% 2536 av dessa kan skriva som en linjärkombination av basmonomen
% 600 av dessa är basmonom och är triviala att skriva om
% Jag gör en matrix wm3 där kolumn j är denna linjär kombination
% som beskriver hur monom j kan skrivas som en linjärkombination
% av de 600 basmonomen.
j1 = 1:2536;
j2 = 2537:size(wm,2);
wm2 = [eye(size(wm,1)) wm(:,j1)\wm(:,j2)];
wm3 = [-wm2(:,j2)' eye(600)];

% Nu tar vi fram variabel 'e' och 'f'
emon = [0 0 0 0 1 0 0 0]';
fmon = [0 0 0 0 0 1 0 0]';

% how are the basis monomials Bv mapped by multiplication with e
Bve = Bv + repmat(emon,1,600);
Bvf = Bv + repmat(fmon,1,600);

% where do these monomials end up?
ind_e = get_index_of_monomials(Bve,wmmon);
ind_f = get_index_of_monomials(Bvf,wmmon);
% read of how the 600 basis monomials in v are expressed 
% as a linear combination of the 600 monomials in w. 
Me = wm3(:,ind_e)';
Mf = wm3(:,ind_f)';

% Solve the generalized eigenvalue problem
[v,d] = eig(Mf,Me);

% Check if ground truth f/e is among the 
% 600 eigenvalues
ddd = diag(d);
gt_fe = gt_sol(6)/gt_sol(5);
disp('tmpv borde bli litet och tmpi talar om vilken av lösningarna som är den vi använde för att generera problemet')
[tmpv,tmpi]=min(abs(diag(d)-gt_fe))
% tmpi talar i så fall om vilken av de 600 lösningarna
% som är den vi använde för att generera problemet. 

%% Varje egenvektor innehåller olika monom av
% lösningspunkten. 
% Det här borde vi kunna göra bättre. 
% Jag undersökte vilka två element kan divideras med varandra 
% för att ta fram en kvot av variablerna (a-h)
xrows = [];
for i = 1:599;
    for j = (i+1):600;
        if sum(abs(Bw(:,i)-Bw(:,j))) == 2,
            %[i j]
            %Bw(:,[i j])
            tmp = Bw(:,i)-Bw(:,j);
            j1 = find(tmp==-1);
            j2 = find(tmp==1);
            xrows = [xrows;[j1 j2 i j]];
        end
    end
end
for i = 1:size(xrows,1);
    if xrows(i,1)>xrows(i,2),
        xrows(i,:) = xrows(i,[2 1 4 3]);
    end
end
xrows = sortrows(xrows);
pairlist = unique(xrows(:,1:2),'rows');
% Varje rad i xrows innehåller
% i1 i2 j1 j2
% så att sol(i1)/sol(i2) = v(j1)/v(j2)
% eller något sånt

% Generera lösningskvoter för alla 600 lösningar
res = zeros(size(pairlist,1),1);
for j = 1:600;
    xx = zeros(size(xrows,1),1);
    for i = 1:size(xrows,1);
        xx(i)=v(xrows(i,3),j)./v(xrows(i,4),j);
    end

    for i = 1:size(pairlist,1);
        blubb  = find((xrows(:,1)==pairlist(i,1) & (xrows(:,2)==pairlist(i,2))));
        res(i,j)=median(xx(blubb));
    end
end;

%[gt_sol(pairlist(:,2))./gt_sol(pairlist(:,1))]

% Borde göra något smartare här tror jag?
% Pallar inte nu

q2 = [ones(1,size(res,2)); res([1 2 3],:)];
q3 = [ones(1,size(res,2)); res([7 8 9],:)];

q2 = q2 ./ repmat(sqrt(sum(q2.^2)),4,1);
q3 = q3 ./ repmat(sqrt(sum(q3.^2)),4,1);

sols = [q2;q3];

% Dessa två borde bli ungefär samma
disp('Om allt gått bra är de två kolumnerna lika')
[sols(:,tmpi) gt_sol]

% Man skulle kunna kolla att fler lösningar uppfyller ekvationerna
disp('Kollar två lösningar till')
evaluate(eqs,sols(:,1))
evaluate(eqs,sols(:,42))

t1 = toc(t0)

% 38 sekunder. Det var 56 sekunder tidigare.
