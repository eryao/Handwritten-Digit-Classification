function ux=u(v)
for i=1:10
    if v(i,:)>=0
        ux(i,:)=1;
    else
        ux(i,:)=0;
    end
end
end

