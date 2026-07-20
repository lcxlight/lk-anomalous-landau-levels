ClearAll["Global`*"];

scriptDir = DirectoryName[$InputFileName];
repoRoot = ParentDirectory[ParentDirectory[scriptDir]];
outFile = FileNameJoin[{ParentDirectory[scriptDir], "deeper-equation-audit-wolfram.out"}];

results = {};
record[name_, expr_] := Module[{ok = TrueQ[FullSimplify[expr, assumptions]]},
  AppendTo[results, {name, ok}];
  Print[name, ": ", If[ok, "PASS", "FAIL"]];
  ok
];
recordRaw[name_, ok_] := Module[{pass = TrueQ[ok]},
  AppendTo[results, {name, pass}];
  Print[name, ": ", If[pass, "PASS", "FAIL"]];
  pass
];

assumptions = Element[{a, c, B, n, Lambda, d, p, kT, v, Ddeg, kx, ky}, Reals] &&
  a > 0 && c > 0 && B > 0 && n >= 1 && Lambda > 0 &&
  p > 0 && kT > 0 && v > 0 && Ddeg > 0;

Print["Deeper equation audit started."];

(* THF model, flat band, and eigenvectors. *)
k2 = kx^2 + ky^2;
kp = kx + I ky;
km = kx - I ky;
h = {{a k2, c kp}, {c km, Delta}};
char = Det[h - x IdentityMatrix[2]];
record["THF characteristic polynomial",
  char == x^2 - (a k2 + Delta) x + (a Delta - c^2) k2];
record["Ideal flat-band factorization",
  (char /. Delta -> c^2/a) - x (x - (a k2 + c^2/a)) == 0];

nrm = a^2 k2 + c^2;
uMinus = {-c, a km}/Sqrt[nrm];
uPlus = {a kp, c}/Sqrt[nrm];
hIdeal = h /. Delta -> c^2/a;
record["Flat-band eigenvector has zero shifted eigenvalue",
  hIdeal.uMinus == {0, 0}];
record["Upper-band eigenvector has eigenvalue a k^2+c^2/a",
  hIdeal.uPlus == (a k2 + c^2/a) uPlus];
record["Bloch states normalized and orthogonal",
  ComplexExpand[Conjugate[uMinus].uMinus] == 1 &&
    ComplexExpand[Conjugate[uPlus].uPlus] == 1 &&
    ComplexExpand[Conjugate[uPlus].uMinus] == 0];

(* Quantum geometry. *)
duMx = D[uMinus, kx];
duMy = D[uMinus, ky];
mx = ComplexExpand[Conjugate[uPlus].duMx];
my = ComplexExpand[Conjugate[uPlus].duMy];
record["Interband matrix elements",
  mx == a c/(a^2 k2 + c^2) && my == -I a c/(a^2 k2 + c^2)];
qxx = ComplexExpand[Conjugate[mx] mx];
qyy = ComplexExpand[Conjugate[my] my];
qxy = ComplexExpand[Conjugate[mx] my];
metric = c^2/(a k2 + c^2/a)^2;
berry = 2 c^2/(a k2 + c^2/a)^2;
record["Quantum metric and Berry curvature",
  qxx == metric && qyy == metric && ComplexExpand[-2 Im[qxy]] == berry];
record["Ideal geometry trace condition",
  2 metric == berry];
Tint = Integrate[(2 metric)/(2 Pi)^2 2 Pi r /. k2 -> r^2, {r, 0, Lambda},
  GenerateConditions -> False];
record["Integrated metric over cutoff disk",
  Tint == a Lambda^2/(2 Pi (a Lambda^2 + c^2/a))];

(* Landau-level block algebra. *)
uN = c^2/a + (2 n + 1) a B;
vN = c^2/a + (2 n - 1) a B;
blockLL = {{(2 n + 1) a B, c Sqrt[2 B n]}, {c Sqrt[2 B n], c^2/a}};
blockTR = {{(2 n - 1) a B, c Sqrt[2 B n]}, {c Sqrt[2 B n], c^2/a}};
record["LL block characteristic equation",
  Det[blockLL - x IdentityMatrix[2]] == x^2 - uN x + c^2 B];
record["LLTR block characteristic equation",
  Det[blockTR - x IdentityMatrix[2]] == x^2 - vN x - c^2 B];

ellM = 1/2 (uN - Sqrt[uN^2 - 4 c^2 B]);
ellTRM = 1/2 (vN - Sqrt[vN^2 + 4 c^2 B]);
ellP = 1/2 (uN + Sqrt[uN^2 - 4 c^2 B]);
ellTRP = 1/2 (vN + Sqrt[vN^2 + 4 c^2 B]);
record["LL branch roots solve LL characteristic",
  (ellM^2 - uN ellM + c^2 B == 0) && (ellP^2 - uN ellP + c^2 B == 0)];
record["LLTR branch roots solve LLTR characteristic",
  (ellTRM^2 - vN ellTRM - c^2 B == 0) && (ellTRP^2 - vN ellTRP - c^2 B == 0)];

(* Gap asymptotics. *)
gTop = -1/2 (a B + c^2/a) + 1/2 Sqrt[(c^2/a)^2 + 2 c^2 B + 9 a^2 B^2];
record["Top-gap low-field asymptote",
  SeriesCoefficient[gTop, {B, 0, 2}] == 2 a^3/c^2];
recordRaw["Top-gap large-field asymptote numeric spot check",
  Abs[N[(gTop - (a B - c^2/(3 a))) /. {a -> 1.115, c -> 0.215, B -> 10^6}]] < 10^-8];

(* Local spacing and effective mass. Use d = mu-E0. *)
uStar = d + c^2 B/d;
vStar = d - c^2 B/d;
dnLL = (uStar - c^2/a)/(2 a B) - 1/2;
dnTR = (vStar - c^2/a)/(2 a B) + 1/2;
record["LL- inversion derivative",
  D[dnLL, d] == 1/(2 a B) - c^2/(2 a d^2)];
record["LLTR- inversion derivative",
  D[dnTR, d] == 1/(2 a B) + c^2/(2 a d^2)];
vLLMinus = 2 a B d^2/(c^2 B - d^2);
vLLTRMinus = 2 a B d^2/(d^2 + c^2 B);
vLLPlus = 2 a B d^2/(d^2 - c^2 B);
vLLTRPlus = 2 a B d^2/(d^2 + c^2 B);
record["Local spacing from inverse slope, LL- and LLTR-",
  FullSimplify[-1/D[dnLL, d] == vLLMinus, assumptions && d > 0 && d^2 < c^2 B] &&
    FullSimplify[1/D[dnTR, d] == vLLTRMinus, assumptions && d < 0]];
record["Effective masses for anomalous branches",
  B/vLLMinus == (c^2 B - d^2)/(2 a d^2) &&
    B/vLLTRMinus == 1/(2 a) + c^2 B/(2 a d^2)];
record["Missing-B LLTR- mass formula is not generally equivalent",
  Not[TrueQ[FullSimplify[1/(2 a) + c^2/(2 a d^2) == B/vLLTRMinus,
    assumptions && d != 0]]]];
record["Upper-branch spacings reduce to parabolic spacing",
  Limit[vLLPlus/(2 a B), d -> Infinity] == 1 &&
    Limit[vLLTRPlus/(2 a B), d -> Infinity] == 1];

(* LK prefactor. The thermal-kernel Fourier integral is covered by
   the author's pre-staging Wolfram figure/equation checks. *)
lambdaDef = 2 Pi p kT/v;
xDef = Pi lambdaDef;
prefactorDerived = FullSimplify[
  (-2 kT Ddeg/v) * kT * (-(Pi lambdaDef/Sinh[Pi lambdaDef])/lambdaDef^2),
  assumptions];
prefactorExpected = Ddeg v/(2 Pi^2 p^2) * (xDef/Sinh[xDef]);
record["LK finite-temperature harmonic prefactor",
  xDef == 2 Pi^2 p kT/v && prefactorDerived == prefactorExpected];

(* Fixed-density derivative cancellation. *)
record["Fixed-density Helmholtz derivative cancellation",
  (((omegaB + omegaMu muB) + muB q) /. omegaMu -> -q) == omegaB];

(* Numeric anchors. *)
aval = 1.115; cval = 0.215; Lval = 0.5;
recordRaw["Bstar numeric anchor",
  Abs[N[cval^2/aval^2] - 0.03718] < 5*10^-5];
recordRaw["Parabolic mass numeric anchor",
  Abs[N[1/(2 aval)] - 0.4484] < 5*10^-4];
recordRaw["Integrated metric numeric anchor",
  Abs[N[(aval Lval^2)/(2 Pi (aval Lval^2 + cval^2/aval))] - 0.13855] < 5*10^-5];

failures = Select[results, ! TrueQ[#[[2]]] &];
Print["Deeper equation audit complete. Checks: ", Length[results],
  ", failures: ", Length[failures]];
If[Length[failures] > 0, Print["Failures: ", failures]];
Export[outFile,
  StringRiffle[
    Join[
      {"Deeper equation audit Wolfram output", "checks=" <> ToString[Length[results]],
        "failures=" <> ToString[Length[failures]]},
      (#[[1]] <> ": " <> If[#[[2]], "PASS", "FAIL"] & /@ results)
    ],
    "\n"
  ],
  "Text"
];
If[Length[failures] > 0, Exit[1], Exit[0]];
