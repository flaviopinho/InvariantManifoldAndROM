classdef MultiIndexFixedOrder
    %Cria uma classe para representar polinomios de determinada ordem
    properties
        %ordem do polinomio
        order
        %Dimensao do vetor de variaveis
        variable_dimension
        %Dimensoes do tensor de saida
        result_shape
        %Ordem do tensor de saida
        result_rank
        %Expoentes dos polinomios
        exponents
        %Dicionario que categoriza os expoentes
        exponents_dict
        %Tensor que armazena os coeficientes dos monomios
        tensor
        % Maximum order of each variable
        maximum_order
    end
    
    methods
        % Cria uma entidade MultiIndexFixedOrder
        function obj = MultiIndexFixedOrder(varargin)
            
            if nargin<2
                error('MultIndex must have at least two argument.');
            end
            
            if nargin==2
                if isa(varargin{1}, 'sptensor') && isvector(varargin{2})
                    obj=MultiIndexFixedOrder.tensor_to_multiindex(varargin{1}, varargin{2});
                    return
                else
                    variable_dimension=varargin{1};
                    order=varargin{2};
                    result_shape=[];
                    maximum_order=[];
                end
            elseif nargin==3
                variable_dimension=varargin{1};
                order=varargin{2};
                result_shape=varargin{3};
                maximum_order=[];
            elseif nargin==4
                variable_dimension=varargin{1};
                order=varargin{2};
                result_shape=varargin{3};
                maximum_order=varargin{4};
            else
                error('invalid input arguments');
            end
            
            if isinteger(variable_dimension) || isinteger(order)
                error('invalid input arguments');
            end
            
            if variable_dimension < 1
                error('variable_dimension must be a positive integer.');
            end
            
            if order<0
                error('Order must be positive or zero.');
            end
            
            if ~isempty(result_shape)
                if ~isnumeric(result_shape) || ~isequal(class(result_shape), 'double') || ~isvector(result_shape) || any(result_shape < 1)
                    error('result_shape must be a vector of positive integers.');
                end
            end
            
            if ~isempty(maximum_order)
                if ~isnumeric(maximum_order) || ~isequal(class(maximum_order), 'double') || ~isvector(maximum_order) || any(maximum_order < 0) || numel(maximum_order)~=variable_dimension
                    error('result_shape must be a vector of positive integers.');
                end
            else
                maximum_order=ones(1, variable_dimension)*order;
            end
            maximum_order(maximum_order<order)=0;
            obj.order=order;
            obj.variable_dimension = variable_dimension;
            obj.result_shape = result_shape;
            obj.result_rank = length(result_shape);
            obj.maximum_order = maximum_order;
            obj.exponents = combine_sum(variable_dimension, order);
            
            %filter
            obj.exponents=obj.exponents(all(obj.exponents <= maximum_order,2), :);
            
            for i=1:size(obj.exponents,1)
                keys{i}=mat2str(obj.exponents(i,:));
                values{i}=i;
            end
            obj.exponents_dict=containers.Map(keys, values);
            obj.tensor = sptensor([], [], [result_shape, size(obj.exponents, 1)]);
        end
        
        function idx=return_index(obj, monomial)
            % Remover linhas repetidas de monomial
            [~, unique_indices, idx_unique] = unique(monomial, 'rows');
            monomial_unique = monomial(unique_indices, :);
            
            % Calcular os índices para as linhas únicas de monomial
            keys = arrayfun(@(x) mat2str(monomial_unique(x,:)), 1:size(monomial_unique, 1), 'UniformOutput', false);
            idx_unique2 = cell2mat(values(obj.exponents_dict, keys))';
            
            % Repetir os índices de acordo com as repetições em monomial
            idx = idx_unique2(idx_unique); % Garante que idx tenha o mesmo número de linhas que monomial
        end
        
        function n=number_of_monomials(obj)
            n=size(obj.exponents, 1);
        end
        
        function obj = add_monomial(obj, monomial_exponents, coefficient, coordinates)
            if size(monomial_exponents,2) ~= obj.variable_dimension
                error('monomial_exponents dimension must be equal to variable_dimension.');
            end
            if ~isnumeric(monomial_exponents) || any(mod(monomial_exponents, 1))
                error('Each element in monomial_exponents must be an integer.');
            end
            if nargin < 4
                coordinates = [];
            end
            if size(coordinates,2) ~= obj.result_rank
                error('Coordinate rank must be the same as result rank.');
            end
            if any(coordinates > obj.result_shape) || any(coordinates <= 0)
                error('Coordinates dont match the result_shape.');
            end
            
            if any(sum(monomial_exponents, 2) ~= obj.order)
                error('monomial order wrong.');
            end
            
            %RowIdx=find(ismember(obj.exponents,
            %monomial_exponents,'rows'), 1);
            
            RowIdx=obj.return_index(monomial_exponents);
            
            if isempty(RowIdx)
                error('monomial wrong.');
            end
            
            coordinates = [coordinates, RowIdx];
            try
                value = obj.tensor(coordinates);
            catch
                value=0;
            end
            obj.tensor(coordinates)=value+coefficient;
        end
        
        function jacob = jacobian(obj)
            jacob_shape = [obj.result_shape, obj.variable_dimension];
            jacob = MultiIndexFixedOrder(obj.variable_dimension, obj.order-1, jacob_shape);
            
            coordinates = obj.tensor.subs;
            values = obj.tensor.vals;
            number_of_coordinates = size(coordinates,1);
            for i = 1:number_of_coordinates
                value=values(i,1);
                coordinate=coordinates(i,:);
                monomial_idx = coordinate(end);
                monomial_exponents = obj.exponents(monomial_idx ,:);
                
                for idx_var = 1:obj.variable_dimension
                    exponent = monomial_exponents(idx_var);
                    if exponent == 0
                        continue;
                    end
                    drop = exponent;
                    monomial_jacob = monomial_exponents;
                    monomial_jacob(idx_var) = drop - 1;
                    jacob_coefficient = value * drop;
                    jacob_coordinate = coordinate;
                    jacob_coordinate(end) = idx_var;
                    jacob=jacob.add_monomial(monomial_jacob, jacob_coefficient, jacob_coordinate);
                end
            end
        end
        
        function result = mtimes(obj, multiplicador)
            if numel(multiplicador)==1
                result =mtimes_number(obj, multiplicador);
                return
            elseif isvector(multiplicador)
                result =mtimes_vector(obj, multiplicador);
                return
            elseif isequal(class(multiplicador), 'MultIndex')
                result =mtimes_multiindex(obj, multiplicador);
                return
            else
                return
            end
        end
        
        function result = plus(obj, A)
            if any(A.result_shape~=obj.result_shape) || A.variable_dimension~=obj.variable_dimension || A.order~=obj.order
                error('Incomplatible MultiIndexFixedOrder to sum');
            end
            result=obj;
            result.tensor=obj.tensor+A.tensor;
        end
        
        function C=multiindex_times_multiindex(A, B, tA, tB, maximum_order)
            tensor_aux=sptensor(ttt(A.tensor, B.tensor, tA, tB));
            shape_a=size(A.tensor);
            shape_a(tA)=[];
            rank_a=numel(shape_a);
            shape_b=size(B.tensor);
            shape_b(tB)=[];
            rank_b=numel(shape_b);
            
            C_result_shape=[shape_a(1:end-1), shape_b(1:end-1)];
            
            pp=[1:rank_a-1, rank_a+1:rank_b-1+rank_a, rank_a, rank_b+rank_a];
            tensor_aux = permute(tensor_aux, pp);
            
            if nargin>4
                C=MultiIndexFixedOrder(A.variable_dimension, A.order+B.order, C_result_shape, maximum_order);
            else
                C=MultiIndexFixedOrder(A.variable_dimension, A.order+B.order, C_result_shape);
            end
            if ~isempty(tensor_aux.subs)
                subsA=tensor_aux.subs(:,end-1);
                subsB=tensor_aux.subs(:,end);
                A_monomials=A.exponents(subsA,:);
                B_monomials=B.exponents(subsB,:);
                C_monomials=A_monomials+B_monomials;
                
                idx_tensor=[1:numel(tensor_aux.vals)]';
                
                if nargin>4
                    idx_tensor = idx_tensor(all(C_monomials <= C.maximum_order,2));
                end
                
                C_idx=C.return_index(C_monomials(idx_tensor,:));
                
                C_subs=[tensor_aux.subs(idx_tensor, 1:end-2), C_idx];
                C_vals=tensor_aux.vals(idx_tensor);
                C.tensor=sptensor(C_subs, C_vals, C.tensor.size);
            else
                C.tensor=sptensor([], [], C.tensor.size);
            end
            
        end
        
        function tensor=convert_to_tensor(obj)
            tensor_shape=[obj.result_shape, ones(1, obj.order)*obj.variable_dimension];
            if ~isempty(obj.tensor.subs)
                tensor_subs1=obj.tensor.subs(:, 1:end-1);
                non_singular_monomials=obj.tensor.subs(:, end);
                tensor_vals=obj.tensor.vals;
                
                index_tensor=obj.exponents_to_index();
                tensor_subs2=index_tensor(non_singular_monomials,:);
                
                tensor_subs=[tensor_subs1, tensor_subs2];
                
                tensor=sptensor(tensor_subs, tensor_vals, tensor_shape);
            else
                tensor=sptensor([], [], tensor_shape);
            end
        end
    end
    methods(Access=private)
        function indices_repetidos=exponents_to_index(obj)
            % Número de linhas na matriz
            
            num_linhas = size(obj.exponents, 1);
            
            % Vetor de índices
            indices = repmat(1:size(obj.exponents, 2), num_linhas, 1);
            
            % Gerar índices repetidos de acordo com os expoentes
            indices_repetidos = cell(num_linhas, 1);
            for i = 1:num_linhas
                indices_repetidos{i} = repelem(indices(i,:), obj.exponents(i,:));
            end
            
            % Converter para matriz
            indices_repetidos = cell2mat(indices_repetidos);
        end
        function result = mtimes_vector(obj, vector_state)
            if ~isnumeric(vector_state) || ~isequal(class(vector_state), 'double') || ~isvector(vector_state) || length(vector_state) ~= obj.variable_dimension
                error('vector_state must be a vector of length equal to variable_dimension.');
            end
            subs = obj.tensor.subs;
            used_exponents = subs(:, end);
            
            [used_exponents_unique, ia, ic] = unique(used_exponents, 'rows');
            
            alpha = obj.exponents(used_exponents_unique, :);
            U_alpha = prod(((vector_state').^alpha)', 1)';
            U_tot=ones(size(obj.exponents,1),1);
            U_tot(used_exponents,1)=U_alpha(ic);
            result = sptensor(tensor(ttv(obj.tensor, U_tot, obj.result_rank+1)));
        end
        function result = mtimes_number(obj, number)
            if ~isnumeric(number)
                error('Must be a number.');
            end
            
            result = obj;
            result.tensor=result.tensor*number;
        end
    end
    methods(Static)
        function tensor_derivative=derivative_tensor(variable_dimension, order, maximum_order)
            maximum_order(maximum_order<order)=0;
            monomials1 = combine_sum(variable_dimension, order);
            if nargin>2
                monomials1=monomials1(all(monomials1 <= maximum_order,2), :);
            end
            
            number_of_monomials=size(monomials1,1);
            tensor_result=[number_of_monomials, variable_dimension];
            tensor_derivative=MultiIndexFixedOrder(variable_dimension, order-1, tensor_result, maximum_order);
            
            monomials=sptensor(monomials1);
            
            
            vals=monomials.vals;
            subs=monomials.subs;
            number_of_exponents = size(subs,1);
            
            monomial_idx=subs(:,1);
            idx_var=subs(:,2);
            exponent = vals(:);
            drop = exponent;
            
            monomial_tensor_derivative = monomials1(monomial_idx,:);
            monomial_tensor_derivative(sub2ind([number_of_exponents, variable_dimension], (1:number_of_exponents)', idx_var)) = drop-1;
            %monomial_tensor_derivative(:,idx_var) = drop - 1;
            tensor_derivative_coefficient = drop;
            
            idx_monomials=tensor_derivative.return_index(monomial_tensor_derivative);
            tensor_derivative_coordinate = [monomial_idx, idx_var, idx_monomials];
            
            tensor_derivative.tensor=sptensor(tensor_derivative_coordinate , tensor_derivative_coefficient , [size(tensor_derivative.tensor)]);
            
        end
    end
    methods(Static, Access=private)
        function result=tensor_to_multiindex(tensor, index_variables)
            
            tensor_rank=numel(size(tensor));
            
            result_index=setdiff(1:tensor_rank, index_variables);
            result_shape=size(tensor);
            result_shape=result_shape(result_index);
            order=numel(index_variables);
            variables_dimension=size(tensor, index_variables(1,1));
            result=MultiIndexFixedOrder(variables_dimension, order, result_shape);
            
            subs=tensor.subs(:, index_variables);
            
            indices = repmat((1:size(subs, 1))', order, 1);
            used_monomials = accumarray([indices(:), subs(:)], 1, [size(subs, 1), max(subs(:))]);
            
            index_monomials=result.return_index(used_monomials);
            subs2=[tensor.subs(:, result_index), index_monomials];
            result.tensor=sptensor(subs2, tensor.vals, [size(result.tensor)]);
        end
    end
end


