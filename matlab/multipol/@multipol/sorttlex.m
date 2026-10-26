function p2 = sorttlex(pp1);
% MULTIPOL/SORT operator
% Sorts the monomials and removes dubble terms
% p2 = sort(p1);
p2=pp1;
for k = 1 : numel(pp1)
    p1 = pp1(k);
    monomials = p1.monomials;
    coeffs = p1.coeffs;
    % Remove double terms if there is more than one term
    if length(coeffs)>1
        temp = [-sum(monomials, 1)' -monomials'];
        [slask,I]=sortrows(temp);
        monomials = monomials(:,I);
        coeffs=coeffs(I);
        ee = find(sum(abs(diff(monomials'))')==0);
        for i=size(ee,2):-1:1;
            coeffs(ee(i))=coeffs(ee(i))+coeffs(ee(i)+1);
        end;
        coeffs(ee+1)=[];
        monomials(ee+1,:)=[];
    end;
    p2(k) = multipol(coeffs,monomials);
end
