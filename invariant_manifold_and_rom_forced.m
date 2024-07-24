function [W, R]=invariant_manifold_and_rom_forced(modes, order, A, B, F, E, Omega, style, tol, tol_load, maximum_order)
% invariant_manifold_and_rom _forced
%% Invariant manifold and ROM
%
%  Author: Flávio Augusto Xavier Carneiro Pinho
%
%  Determination of the invariant manifold (W) and the Reduced Order Model
%  (R) of the forced system:
%
% $$B_{ij}\dot{z}_j=A_{ij}z_j+F^2_{ijk}z_j z_k+F^3_{ijk}z_j z_k z_l+\dots+E^+_ip_++E^-_ip_-$$
%
% $$\mathbf{p}=[p_1, \dots, p_d, p_+, p_-]^T$$
%
% $$\dot{p}_+=i\Omega p_+$$
%
% $$\dot{p}_-=-i\Omega p_-$$
%
% Based on the papers:
%
% Vizzaccaro, A., Gobat, G., Frangi, A. et al.
% Direct parametrisation of invariant manifolds for non-autonomous
% forced systems including superharmonic resonances.
% Nonlinear Dyn 112, 6255–6290 (2024).
% https://doi.org/10.1007/s11071-024-09333-0
%
% Jain, S., Haller, G. How to compute invariant manifolds and their reduced
% dynamics in high-dimensional finite element models.
% Nonlinear Dyn 107, 1417–1450 (2022).
% https://doi.org/10.1007/s11071-021-06957-4
%
% Obs.: This code uses Tensor Toolbox:
%        Brett W. Bader, Tamara G. Kolda and others,
%        Tensor Toolbox for MATLAB, Version 3.6,
%        https://www.tensortoolbox.org
%        September 28, 2023.
%
% Obs2.: See implematation of MultiIndexFixedOrder
%
% Last update: 23-05-2024
%
% *Inputs* :
%
% * $\mathbf{A}$: square matrix or sparse tensor (sptensor)
% * $\mathbf{B}$: square matrix or sparse tensor (sptensor)
% * $\mathbf{F}$: Each cell must contain a sparse tensor or the same order
% of the cell index
% * $\mathbf{E}$: Matrix that contains the vectors $E^+$ and $E^-$
% * $\Omega$: Frequency of the external load
% * |modes| : index of the master modes
% * |order| : maximum order for parametrization of the manifold and rom
% * |tol| : tolerance to detect inner resonances
% * |tol_load| : tolerance to detect resonances with the external load
% * |maximum_order| : (optional) maximum order of each variables of the
% parametrizations of the manifold and rom
%
% *Outputs* :
%
% * $\mathbf{R}$ : ROM (Represented as MultiIndexFixedOrder)
% * $\mathbf{W}$ : Parametrization of the manifold
% (Represented as MultiIndexFixedOrder)
%
% $$z_j=W^r_{jI}\ p^{\mathbf{\alpha}^r_I}$$
%
% $$\dot{p}_a=R^s_{aJ}\ p^{\mathbf{\alpha}^s_J}$$
%
% $$p^{\mathbf{\alpha}^s_J}=\prod_{a=1}^Mp^{\alpha^s_{Ja}}_a$$

addpath('C:\Matlab\tensor_toolbox\');

% Dimension of the state variables.
N=size(A, 1);

% Dimension of the ROM.
% The last two modes represent the external load.
d=length(modes);
M=d+2;

% Transform F into a uppter triangular tensor
F=optimize_f(F);

%% Solution for order = 1
%
% Eigenproblem of the linear system
%
% $$\bar{U}_{ik}B_{im}=\bar{U}_{ik}A_{im}/\lambda^k$$
%
% $$B_{im} V_{mj}=A_{im}V_{mj}/\lambda^j $$
%
% $$\bar{U}_{ik} B_{im} V_{m}=\delta_{jk}$$
%
% * $\mathbf{U}$: Left eigenvectors
% * $\mathbf{V}$: Right eigenvectors
% * $\mathbf{\Lambda}$: Eigenvalues
% * _E_ index: Eingenspace of the master modes

[V0,Lambda0,U0]=eig(A,B);
[~,ind]=sort(abs(imag(diag(Lambda0))));
ind_zero=find(abs(imag(diag(Lambda0)))==0);
ind=[setdiff(ind, ind_zero,'stable'); ind_zero];
V0=V0(:,ind);
U0=U0(:,ind);
Lambda0=Lambda0(ind,ind);

aux = diag(U0'*B*V0);

U0 = U0*diag(1./(aux'));

external=setdiff(1:N, modes);
Lambda_external=Lambda0(external,external);
Lambda_E=Lambda0(modes, modes);

Lambda_E=diag(Lambda_E);
Lambda_external=diag(Lambda_external);

U_E=U0(:, modes);
V_E=V0(:, modes);

lambda_Load=[1i*Omega; -1i*Omega];

% Resonances with external load.
% Assuming that the external load can be resonant with any linear
% frequency of the mechanical model
tolerances = abs((Lambda_E - lambda_Load.'))./abs(Lambda_E(1));
cond=tolerances < tol_load;
[res1, res2]=find(cond);
fprintf('%i resonances with external load. \n', numel(res1));


W_load=zeros(N, 2);
R_load=zeros(M, 2);

for i=1:2
    res=res1(res2==i);
    n_res=numel(res);
    
    Maux=[B*lambda_Load(i)-A,         B*V_E(:, res),     zeros(N, M-n_res);
               U_E(:,res)'*B,   zeros(n_res, n_res), zeros(n_res, M-n_res);
           zeros(M-n_res, N), zeros(M-n_res, n_res),  eye(M-n_res,M-n_res);];
    
    Vaux=[E(:,i);
           zeros(M, 1)];
    X=Maux\Vaux;
    W_load(:,i)=X(1:N);
    R_load(res,i)=X(N+1:N+n_res); 
    R_load(d+i, i)=lambda_Load(i);
end

R1=[[diag(Lambda_E); zeros(2, d)], R_load];
W1=[V_E W_load];

R{1}=MultiIndexFixedOrder(M, 1, M, maximum_order);
W{1}=MultiIndexFixedOrder(M, 1, N, maximum_order);
R{1}.tensor=sptensor(sparse(R1));
W{1}.tensor=sptensor(sparse(W1));

A=sparse(A);
B=sparse(B);

%% Eigenproblem of R{1}
% Used to determine the kernel of L.
[q, mu]=eig(full(R1));
mu=diag(mu);

%% Derivative tensor
%
% dW: _MultiIndexFixedOrder to compute de gradient of z with respect of p_
%
% $$\frac{\partial {z}_j}{\partial {p}_a}=W^r_{jI}\ \partial W^{r-1}_{IaJ}\ p^{\mathbf{\alpha}^{r-1}_J}$$
%

dW0=MultiIndexFixedOrder.derivative_tensor(M, 1, maximum_order);
for o=2:order
    dW{o-1}=MultiIndexFixedOrder.derivative_tensor(M, o, maximum_order);
end

%% Solution of the Invariance equation for o>1
% $$L^o_{imNP}W^o_{mN}\ p^{\mathbf{\alpha}^o_P}=D^o_{icMP}R^o_{cM}\ p^{\mathbf{\alpha}^o_P}+C^o_{iP}\ p^{\mathbf{\alpha}^o_P}$$

% D is independent of o.
D=D_matrix(B, W1);

for o=2:order
    
    W{o}=MultiIndexFixedOrder(M, o, N, maximum_order);
    R{o}=MultiIndexFixedOrder(M, o, M, maximum_order);
    
    n_monomials=size(W{o}.exponents,1);
    
    fprintf('order=%i   (n_monomials=%i)\n', o, n_monomials);
    
    G=G_MultiIndex(dW, R{1}, o, maximum_order);
    G_matrix=spmatrix(G.tensor);
    
    fprintf('   determining C1... ');
    % C1: Compute nonlinear stiffness terms
    C1 = C1_matrix_opt(F, W, M, N, o, maximum_order);
    fprintf('OK\n');
    
    fprintf('   determining C2... ');
    % C2: Remaing terms that depends on B and low orders of W and R
    C2 = C2_matrix(B, W, dW, R, M, N, o, maximum_order);
    fprintf('OK\n');

    C=C1+C2;
    
    fprintf('   determining R... ');
    
    Ro=zeros(M, n_monomials);
    if strcmp(style,'normal form')
        fprintf('   determining kernel of L... ');
        [res1, res2]=null_of_L(M, N, G, q, Lambda_E, Lambda_external, U_E, tol);

    elseif strcmp(style,'graph')
        res1=repmat([1:M-2], 1, n_monomials);
        res2=kron([1:n_monomials], ones(1,M-2));
    else
        error('invalid style');
    end
    fprintf('OK\n');
    
    
    fprintf('   determining W... ');
    
    Wo=zeros(N, n_monomials);
    
    % W can be solve indepently for each monomial.
    % (see appendix B of
    % https://doi.org/10.1007/s11071-024-09333-0)
    
    for i=1:n_monomials
        res=res1(res2==i);
        n_res=numel(res);
        
        L_ii = L_matrix(B, A, G_matrix(i,i), i, i);
        
        % Since R1 is upper triangular, G is also upper triangular,
        % making L block triangular. So the system can be solved
        % iteratively. That is, the solution of W(:,i) depends on W(:,j)
        % for j<i.
        
        h_ij=zeros(N, 1);
        for j=1:i-1
            if(G_matrix(j,i)~=0)
                L_ij = L_matrix(B, A, G_matrix(j,i), i, j);
                h_ij=h_ij-L_ij*Wo(:, j);
            end
        end
        
        % Bordering technique (see
        % https://doi.org/10.1016/j.cma.2021.113957)
        Mat=[              L_ii,         B*V_E(:, res),     zeros(N, M-n_res);
             U_E(:,res)'*B,   zeros(n_res, n_res), zeros(n_res, M-n_res);
            zeros(M-n_res, N), zeros(M-n_res, n_res),  eye(M-n_res,M-n_res);];
        Vec=[ C(:,i)+h_ij;
              zeros(M, 1);];
        
        X=Mat\Vec;
        Wo(:,i)=X(1:N,1);
        Ro(res,i)=X(N+1:N+n_res,1);
        teste=L_ii*Wo(:, i)-(C(:,i)+D*Ro(:,i)+h_ij);
        
    end
    fprintf('OK\n');
    
    R{o}.tensor=sptensor(Ro);
    W{o}.tensor=sptensor(Wo);
end

end

%% Determination of G
% $$G^{o}_{NP}\ p^{\mathbf{\alpha}^{o}_{P}}=G^{o}_{N(JK)}\ p^{\mathbf{\alpha}^{o}_{(JK)}}=\partial W^{o-1}_{NaJ}\ R^1_{aK}\ p^{\mathbf{\alpha}^{o-1}_J+{\mathbf{\alpha}^1_K}}$$
%
% $$P=(JK)$$
function G=G_MultiIndex(dW, R1, k, maximum_order)
G=dW{k-1}.multiindex_times_multiindex(R1, 2, 1, maximum_order);
end

%% Determination of L
% $$L^o_{imNP}=B_{im} G^{o}_{NP}-A_{im}\delta_{NP}$$
function L=L_matrix(B, A, g, i, j)
L=B*g-A*(i==j);
end

%% Determination of C1
% $$C^o_{iP}\ p^{\mathbf{\alpha}^{o}_P}=-\sum_{r=2}^{o-1}B_{ij}W^r_{jI}\ \partial W^{r-1}_{IaJ}\ R^{o-r+1}_{aK}\ p^{\mathbf{\alpha}^{r-1}_J}p^{\mathbf{\alpha}^{o-r+1}_K}$$
function result=C1_matrix(F, W, M, N, o, maximum_order)
result=MultiIndexFixedOrder(M, o, N, maximum_order);
result=result.tensor;

for d=2:numel(F)
    combs=permn([1:o], d);
    indices=sum(combs,2)==o;
    combs=combs(indices,:);
    
    F_tensor=F{d};
    Z=MultiIndexFixedOrder(M, o, N, maximum_order);
    if ~isempty(combs)
        for e=1:size(combs,1)
            W1=W{combs(e,1)};
            F_times_W1=ttt(F_tensor, W1.tensor, 2, 1);
            Z_aux=MultiIndexFixedOrder(W1.variable_dimension, W1.order, F_times_W1.size(1:end-1));
            Z_aux.tensor=F_times_W1;
            for n=2:size(combs,2)
                z=combs(e,n);
                Z_aux=Z_aux.multiindex_times_multiindex(W{z}, 2, 1, maximum_order);
            end
            Z.tensor=Z.tensor+Z_aux.tensor;
        end
    end
    
    result=result+Z.tensor;
end
result=spmatrix(result);
end

% optimized version of the function C1_matrix
function result=C1_matrix_opt(F, W, M, N, o, maximum_order)

Z=MultiIndexFixedOrder(M, o, N, maximum_order);
result=sparse(N, Z.number_of_monomials());

for d=2:numel(F)
    combs=permn([1:o], d);
    indices=sum(combs,2)==o;
    combs=combs(indices,:);
    
    F_tensor=F{d};
    
    if ~isempty(combs)
        for e=1:size(combs,1)
            for n=1:size(combs,2)
                z=combs(e,n);
                WW{n}=spmatrix(W{z}.tensor).';
            end
            F_times_W=ttm(F_tensor, WW, [1:size(combs,2)]+1);
            
            [subs, ~, idx_subs1c]=unique(F_times_W.subs(:,2:end),'rows', 'stable');
            
            W1=W{combs(e,1)};
            Z_monomials=W1.exponents(subs(:, 1),:);
            for n=2:size(combs,2)
                z=combs(e,n);
                Z_monomials=Z_monomials+W{z}.exponents(subs(:, n),:);
            end
            
            Z_idx=zeros(size(subs,1),1);
            minor_than_maximum=find(all(Z_monomials <= Z.maximum_order,2));
            Z_idx(minor_than_maximum,1)=Z.return_index(Z_monomials(minor_than_maximum,:));
            Z_idx=Z_idx(idx_subs1c,1);
            
            filter_zeros=find(Z_idx~=0);
            
            Z_subs=[F_times_W.subs(filter_zeros, 1), Z_idx(filter_zeros,1)];
            Z_vals=F_times_W.vals(filter_zeros,1);
            result=result+sparse(Z_subs(:,1), Z_subs(:,2), Z_vals, Z.tensor.size(1), Z.tensor.size(2));
        end
    end
end
end

%% Determination of C2
% $$C^o_{iP}p^{\mathbf{\alpha}^o_P}=\sum_{o=r+s}F^2_{ijk} W^r_{jI} W^s_{kJ}p^{\mathbf{\alpha}^r_I} p^{\mathbf{\alpha}^s_J}+\sum_{o=r+s+t}F^3_{ijkl} W^r_{jI} W^r_{kJ} W^t_{lK}p^{\mathbf{\alpha}^r_I} p^{\mathbf{\alpha}^s_J}p^{\mathbf{\alpha}^t_K}+\dots$$

function result=C2_matrix(B, W, dW, R, M, N, o, maximum_order)
result=MultiIndexFixedOrder(M, o, N, maximum_order);
result=result.tensor;
for d=2:o-1
    Wj=W{d};
    G=dW{d-1}.multiindex_times_multiindex(R{o-(d-1)},2,1, maximum_order);
    result=result+ttt(Wj.tensor, G.tensor, 2, 1);
end
result=-B*double(result);
end

%% Determination of D
% $$D^o_{icMP}=-B_{ij}W^1_{jc}\delta_{MP}$$

function D=D_matrix(B, W1)
D=-B*W1;
end

%% Determination of near inner ressonances
%
% $$L^o_{imNP}\ \bar{U}_{i\beta}\ q_{P\theta}=0$$
%
% $$N^o_{iP(\beta\theta)}=\bar{U}_{i\beta}\ q_{P\theta}$$
%
% $$G^{o}_{NP}\ q_{PQ}=\mu^Q\ \delta_{NP}\ q_{PQ}$$
%
% $$\bar{U}_{ik}B_{im}=\bar{U}_{ik}A_{im}/\lambda^k$$
%
% for: 
%
% $$\mu^Q/\Lambda^k=1$$

function [rows, cols, No]=null_of_L(M, N, G, q, Lambda_AB_E, Lambda_AB_external, U_AB_E, tol)
% M: dimension of the reduced order model
% N: dimension of the full model
% G: MultiIndex of order o that contains the monomials of the reduced
% order model and depends on R{1}
% q, mu: Right eigenvector and eigenvalues of R{1}
% U_AB, Lambda_AB: Left eigenvector and eigenvalues of matrices A and B
% index E: Index of the master modes
% index external: modes that are not in E
% tol: tolerance considered to capture de resonance

%% Eigenvalues of G
% Since G is a upper triangular matrix, the eigenvalues are given by its
% diagonal.
sigma=diag(double(G.tensor));

%% Finding inner and cross ressonances

% Determining near resonanced
tolerances = abs((Lambda_AB_E - sigma.'))./abs(Lambda_AB_E(1));
cond=tolerances < tol;
[rows, cols] = find(cond);

fprintf('\n   %i inner ressonances found... ', numel(rows));

for m=1:N-M
    indices=find(abs(sigma-Lambda_AB_external(m))<1e-4);
    if(numel(indices)>0)
        fprintf('cross ressonance detected.');
    end
end

if nargout==3
    
    No=[];
    if numel(rows)==0
        return;
    end
    
    %% Eigenvectors of G
    % Determined only for resonant modes
    % (Dont know how it works, but it works)
    
    Q=sparse(n_monomials,n_monomials);
    q=sptensor(q);
    
    ressonant_exponents=unique(cols);
    n_ressonant_exponents=numel(ressonant_exponents);
    
    ex1=To.exponents;
    ex1=prod(factorial(ex1), 2);
    
    for ii=1:n_ressonant_exponents
        i=ressonant_exponents(ii);
        
        monomial=To.exponents(i,:);
        
        idx=[];
        for j=1:M
            idx=[idx, repmat(j, 1, monomial(j))];
        end
        Qi=sptensor(q(:, idx(1)));
        for j=2:To.order
            Qj=sptensor(q(:, idx(j)));
            Qi=ttt(Qi, Qj);
        end
        Qi=sptensor(Qi);
        if nnz(Qi)~=0
            indices = repmat((1:size(Qi.subs, 1))', To.order, 1);
            used_monomials = accumarray([indices(:), Qi.subs(:)], 1, [size(Qi.subs, 1), max(Qi.subs(:))]);
            
            if(size(used_monomials, 2)<M)
                used_monomials(1,M)=0;
            end
            index_monomials=To.return_index(used_monomials);
            qaux=double(sptensor(index_monomials, Qi.vals, [n_monomials]));
            
            Q(:, i)=qaux.*ex1;
        end
    end
    
    Q=Q./max(Q);
    
    %% Determining kernel of L
    U_AB_E=sptensor(conj(sparse(U_AB_E)));
    Q=sptensor(Q);
    
    No=sptensor([], [], [N, n_monomials, numel(rows)]);
    for i=1:numel(rows)
        no=ttt(U_AB_E(:, rows(i)),  Q(:, cols(i)), [],[]);
        No(:, :, i)=no;
    end
    
end

end

function F2=optimize_f(F)

for i=2:numel(F)
    subs=F{i}.subs;
    vals=F{i}.vals;
    size=F{i}.size;

    subs=[subs(:,1), sort(subs(:,2:end), 2)];
    
    F2{i}=sptensor(subs, vals, size);
end

end

