function vec=combine_sum(M, S)
    if M==1
        vec=S;
        return
    end
        
    vec=zeros(1,M);
    vec(1,1)=S;
    for i=1:S
        subvec=combine_sum(M-1, i);
        linhas=size(subvec, 1);
        vec2 = [ones(linhas,1)*(S-i) subvec];
        vec =cat(1, vec, vec2);
    end
end

