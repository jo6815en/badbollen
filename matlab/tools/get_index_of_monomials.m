function indices = get_index_of_monomials(monomials1,monomials2);

nn = size(monomials1,2);
indices = zeros(1,nn);
for i = 1:nn
    indices(i) = find(all(monomials2 == repmat(monomials1(:,i),1,size(monomials2,2))));
end
