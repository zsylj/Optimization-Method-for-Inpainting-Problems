function [PSNR, SSIM, PSNR0, SSIM0, u, out_img] = SB_Inpainting_color_ATV_single(img,img_masked,mask,mu,lambda,tol)
%f: measurement (vector)
%K: linear transformation (matrix)
%u: ground truth (vector)
%psi: Penaltly (matrix) -> D
%u = argmin_u mu/2 ||K*u-f||_2^2 + |psi*u|_1
%Notice: * is matrix multiplication

[img_h, img_w, img_c] = size(img_masked);
N=img_h*img_w;

f=reshape(img_masked,[N*img_c,1]);

[D_c, Dt_c, DtD_c] = DiffOper(img_h, img_w, img_c);
[~, Kt_c, KtK_c] = MtxOper(mask);

b=zeros(2*N*img_c,1);
d=zeros(2*N*img_c,1);%or d=b anyway just init

u=zeros(img_c*N,1);%or u=f anyway just init

err=100;%anyway just init>tol

k=1;%init

while err>tol && k<=100
    %fprintf('iter. %g ',k);
    u_prev = u;%init

%     disp(size(mu))
%     disp(size(KtK_c))
%     disp(size(lambda))
%     disp(size(DtD_c))
%     disp(size(Kt_c))
%     disp(size(f))
%     disp(size(Dt_c))
%     disp(size(d))
%     disp(size(b))

    [u,~] = cgs(mu.*KtK_c+lambda.*DtD_c, mu.*Kt_c*f+lambda.*Dt_c*(d-b),1e-3,100);%step1

    err = norm(u_prev-u)/norm(u);
    %fprintf('err=%g \n',err);

    out_img = reshape(u,[img_h, img_w, img_c]);

    [PSNR0, SSIM0] = standar(img_masked, img);
    [PSNR, SSIM] = standar(out_img, img);
   
    %Shrinkage
    d = shrink(D_c, u, b, lambda);

    %Dub = D_c * u + b;
    %d = wthresh(Dub, 1./lambda);%step2

    b = b + D_c * u - d;%step3

    k = k+1;
end
if k>100
    fprintf('Abort, Err= %g \t',err);
    fprintf('Iter = %g \n', k)
else
    fprintf('Err= %g \t',err);
    fprintf('Iter = %g \n', k)
end
end

function [D_c, Dt_c, DtD_c] = DiffOper(img_h, img_w, img_c)%c stand for color
N=img_h*img_w;
D_v_c = spdiags([-ones(img_c*N,1), ones(img_c*N,1)], [0, 1], img_c*N, img_c*N);%vertical diff mtx (fwd)
D_h_c = spdiags([-ones(img_c*N,1), ones(img_c*N,1)], [0, img_h], img_c*N, img_c*N);%horizontal diff mtx (fwd)
D_c = [D_v_c; D_h_c];
Dt_c = D_c';
DtD_c = Dt_c * D_c;
end

function [K_c, Kt_c, KtK_c] = MtxOper(mask)
[mask_h, mask_w, mask_c] = size(mask);
N=mask_h*mask_w;
mask_v=reshape(mask(:,:,:),[mask_c*N,1]);
K_c=spdiags([mask_v], 0, mask_c*N, mask_c*N);
Kt_c = K_c';
KtK_c = Kt_c * K_c;
end

function [PSNR, SSIM] = standar(out, ref)
PSNR = psnr(out, ref);
SSIM = ssim(out, ref);
end
