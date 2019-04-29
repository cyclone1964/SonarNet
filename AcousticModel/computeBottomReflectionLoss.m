%computeBottomReflectionLoss - compute bottom scattering strength
%
% Loss = computeBottomReflectionLoss(Frequency, Angle, Environment)
%
% computes the bottom reflection loss for the given frequency and
% angle. It supports multiple frequencies and angles: different
% frequencies are returned in different columnns, different angles in
% different rows.
%
% This is currently an implementation of the APL/UW 9407 algorithm,
% ported from the FORTRAN function elblos in CASS V4.2
%
% Copywrite 2010 BBN Technologies, Matt Daily author
function Loss = ...
    computeBottomReflectionLoss(Frequency, Angles, Environment)

% Get the parameters needed
[Rho, Nu, Delta, W2, Gamma, Sigma2, NuT, DeltaT] = ...
    getBottomParameters(Environment.Bottom.GrainSize);

% Check the inputs
if (min(Angles) < 0 | Rho < 1 | Nu <= 0 | Delta < 0 | NuT < 0 | DeltaT < 0)
  error 'Bad Inputs';
end

% This section taken from the FORTRAN function brlael.f in CASS
% V4.2

Al = complex(2*pi/Nu,0);
Kl = Al*complex(1,-Delta);

At = complex(2*pi/NuT,0);
Kt = At*complex(1,-DeltaT);

Loss = ones(size(Angles));
Indices = find(Angles >= 0.015*pi/180);
Kx = 2*pi*cos(Angles);
Kti = abs(Kx);
Ref = computeGamma(Kti,2*pi,Kl,Kt,Rho);
Loss(Indices) = real(Ref(Indices).*conj(Ref(Indices)));

% Convert to dB
Loss = 10 * log10(max(1e-10,Loss));

%computeGamma - compute gamma function for elastic reflection
%
% Gamma = computeGamma(Kx, Kw, Kl, Kt, Rho) computes the complex
% reflection coefficient given the input parameters. The only input
% that can be a vector is the first one.
%
function Gamma = computeGamma(Kx, Kw, Kl, Kt, Rho);

C1 = complex(1,0);
C2 = complex(2,0);
Ckx = complex(Kx, 0);

% Sine and cosine of twice shear wave grazing angle
Sin2Sq = (C1 - C2 * (Ckx/Kt).*(Ckx/Kt)) .^ 2;
Cos2Sq = C1 - Sin2Sq;

% Impedances
CRho = complex(Rho,0);
CKw = complex(Kw, 0);

Zw = C1 ./(Kw * computeBeta(Kx, CKw));
Z1 = CRho ./ (Kl * computeBeta(Kx, Kl));
Zt = CRho ./ (Kt * computeBeta(Kx, Kt));

% Impendance Ratio
Z = (Sin2Sq .* Z1 + Cos2Sq .* Zt) ./ Zw;
Gamma = (Z - C1) ./ (Z + C1);

%computeBeta - compute the beta function for the reflection model
%
% Beta = computeBeta(Kw, K) computes something that I don't
% understand
function Beta = computeBeta(Kx, K)

KxOverK = Kx / K;
Beta = sqrt(1 - KxOverK.*KxOverK);

%getBottomParameters - get the bottom parameters from grain size
%
% [Rho, Nu, Delta, W2, Gamma, Sigma2, NuT, DeltaT] = ...
%    getBottomParameters(GrainSize)
%
% Returns the bottom parameters for the given grain size. This
% ported from the FORTRAN function gsiz2des in Cass V4.2
%
% In case anybody cares, the meaning of these parameters is
%
% Rho - Density ratio
% Nu - Compressional Sound Speed Ratio
% Delta - Shear Loss Parameter
% W2 - Spectral Strength
% Gamma - Spectral Exponent
% Sigma2 - Volume Parameter
% NuT - Compressional Sound Speed Ratio
% DeltaT - Shear Loss Parameter
%
% For more information read APL/UW TR 9407 and GABIM
function [Rho, ...
	  Nu, ...
	  Delta, ...
	  W2, ...
	  Gamma, ...
	  Sigma2, ...
	  NuT, ...
	  DeltaT] = ...
    getBottomParameters(GrainSize)

% Apparently some reference sound speed
Cref = 1528;

if (GrainSize > 9 | GrainSize < -10)
  error 'Bad Grain Size'
end

% Set Rho and Nu
if (GrainSize >= -10 & GrainSize < -7)
  Rho = 2.5;
  Nu = -2.5*(GrainSize+7)/3+1.8*(GrainSize+10)/3.0;
elseif (GrainSize >= -7 & GrainSize < -1)
  Rho = -2.5*(GrainSize+1)/6+2.492*(GrainSize+7)/6.0;
  Nu = -1.8*(GrainSize+1)/6+1.3370*(GrainSize+7)/6.0;
elseif (GrainSize >= -1 & GrainSize < 1.0)
  Rho = 2.3139 + GrainSize * (-0.17057  + GrainSize * 0.007797);
  Nu  = 1.2778 + GrainSize * (-0.056452 + GrainSize * 0.002709);
elseif (GrainSize >= 1.0 & GrainSize < 5.3) 
  Rho = 3.0455 + ...
	GrainSize * (-1.1069031 + ...
		     GrainSize * (0.2290201 - ...
				  GrainSize * 0.0165406));
  Nu  = 1.3425 + ...
	GrainSize * (-0.1382798 + ...
		     GrainSize * (0.0213937 - ...
				  GrainSize * 0.0014881));
elseif (GrainSize >= 5.3 & GrainSize <= 9.0) 
  Rho = 1.1565 - GrainSize * 0.0012973;
  Nu  = 1.0019 - GrainSize * 0.0024324;
end

% Set the Delta
if ( GrainSize < -1 & GrainSize >= -10)
 Delta = (0.0170544*(GrainSize+10)-0.05*(GrainSize+1))/9;
elseif (GrainSize >= -1 & GrainSize < 9.5)
 if (GrainSize >= -1 & GrainSize < 0)
  Kh = 0.4556;
 elseif (GrainSize >= 0 & GrainSize < 2.6)
  Kh = 0.4556+0.0245 * GrainSize;
 elseif (GrainSize >= 2.6 & GrainSize < 4.5)
  Kh = 0.1978+0.1245*GrainSize;
 elseif (GrainSize >= 4.5 & GrainSize < 6.0)
  Kh = 8.0399-2.5228*GrainSize+0.20098*GrainSize^2;
 elseif (GrainSize >= 6.0 & GrainSize < 9.5)
  Kh = 0.9431-0.2041*GrainSize+0.0117*GrainSize^2;
 else
  Kh = 0.0601
 end
 Delta = Kh*Nu*Cref*log(10.0)/(40000*pi);
end

% Now set the NuT
if (GrainSize < -8)
 NuT = 1.3;
elseif (GrainSize >= -8 & GrainSize < -2)
 NuT = (1.1*(GrainSize+8)-1.3*(GrainSize+2))/6;
elseif (GrainSize >= -2 & GrainSize < 1.5)
 NuT = (0.02*(GrainSize+2)-0.3*(GrainSize-1.5))/3.5;
else
 NuT = 0.02;
end

% Set DeltaT
if (GrainSize < -8)
 DeltaT = 0.05;
elseif (GrainSize >= -8 & GrainSize < -6)
 DeltaT =(-0.05*(GrainSize +6)+0.02*(GrainSize+8))/2;
else
 DeltaT = 0.02;
end

% Check something
PTest = (Nu/NuT)^2*(Delta/DeltaT);
if (abs(PTest) < 4/3)
 error 'Bad PTest'
end

% Compute Gamma using eq 8 from APL-UW 9407
Gamma = 3.25;

% Compute rms relief using eq 9 from APL-UW 9407.
% GrainSize=-10 is for rough rock, and GrainSize=-9.9999 is for rock.
% Do linear interpolation to get w2 values for negativa
% grain size values.
HoH0 = (2.03846 + 0.26923) / (1.0 - 0.076923);
W2n1 = 0.00207 * HoH0^2;
W2n7 = 0.016;
W2n99 = 0.01862;

if (GrainSize < -9.9999)
  W2=0.20693;
elseif (GrainSize >= -9.9999 & GrainSize < -7)
  W2=((W2n99-W2n7)/(3)*(abs(GrainSize)-7))+W2n7;
elseif(GrainSize >= -7 & GrainSize < -1)
  W2=((W2n7-W2n1)/(7.-1.)*(abs(GrainSize)-1))+W2n1;
elseif (GrainSize >= -1.0 & GrainSize < 5.0)
  % Compute W2 using eqs 9 and 10 from APL-UW 9407
  HoH0 = (2.03846 - 0.26923 * GrainSize) / (1.0 + 0.076923 * GrainSize);
  W2 = 0.00207 * HoH0^2;
elseif (GrainSize >= 5.0 & GrainSize <= 9.0)
  HoH0 = 0.5;
  W2 = 0.00207 * HoH0^2;
end

% Added by kym to units of meters
H0 = 0.01;
W2m  = W2 *(H0^(-Gamma))*1.e-8;

% Compute sigma2 using eq 6 from APL-UW 9407
if (GrainSize < 5.5)
  Sigma2 = 0.002;
elseif (GrainSize >= 5.5 & GrainSize <= 9.0)
  Sigma2 = 0.001;
end
