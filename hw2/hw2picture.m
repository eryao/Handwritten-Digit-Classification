%{
x2=[0 911;1 127;2 140;3 120;4 69;5 59;6 62;7 59;8 33;9 54;10 48;11 30;12 41;13 16;14 46;15 22;16 15;17 19;18 26;19 9;20 14;21 14;22 26;23 10;24 11;25 5;26 1;27 7;28 6;29 2;30 4;31 4;32 3;33 1;34 1;35 2;36 1;37 0]
 x3=[0 49263;
    1 9258;
    2 9083;
    3 8176;
    4 8451;
    5 8069;
    6 9160;
    7 7858;
    8 7309;
    9 6769;
    10 7806;
    11 7493;
    12 7670;
    13 7845;
    14 7378;
    15 7029;
    16 8397;
    17 7148;
    18 8271;
    19 7353;
    20 8104;
    21 7189;
    22 7072;
    23 7100;
    24 7070]


figure(1)
plot(x1(:,1),x1(:,2),'r')

 xlabel('@ epoch');
    ylabel('# errors');
    title('@ epoch vs # errors')
    figure(2)
plot(x2(:,1),x2(:,2),'r')

%hold on
 xlabel('@ epoch');
    ylabel('# errors');
    title('@ epoch vs # errors')
%}
%(d)
trainImages = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
elta=1;
n=60000;
e=0.2;%when n=60000,set e=0.135
trainimages=trainImages(:,1:n);
trainlabels=trainLabels(1:n);
W=(1-(-1)).*rand(10,784)+(-1);
epoch=0;
errors=100000;%when n=60000,set errors=100000
while ((errors/n)>e)
    errors=0;
    for i=1:n
        x=trainimages(:,i);
        v=W*x;
        vj=max(v);
        for r=1:10
            if v(r,:)==vj
                j=r-1;
            end
        end
 if j==trainlabels(i)
            errors=errors;
        else
            errors=errors+1;
        end
    end
    epoch=epoch+1;
    for i=1:n
        dx=zeros(10,1);
        x=trainimages(:,i);
        label=trainlabels(i);
        dx(label+1,:)=1;
        v=W*x;
        ux=u(v);
        W=W+elta*(dx-ux)*x';
    end
    N(epoch,1)=epoch-1;
    N(epoch,2)=errors;
    N1=N(:,1);
    N2=N(:,2);
    plot(N1,N2,'r')
    hold on
 xlabel('@ epoch');
    ylabel('# errors');
    title('@ epoch vs # errors')
end
%(e)
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
errors_test=0;
for i=1:10000
    xtest=testImages(:,i);
    vtest=W*xtest;
    vjtest=max(vtest);
    for r=1:10
        if vtest(r,:)==vjtest
            jtest=r-1;
        end
    end
    if jtest==testLabels(i)
        errors_test=errors_test;
    else
        errors_test=errors_test+1;
    end
end
per=errors_test
