ClearAll["Global`*"];

(* Focused WolframScript audit for appendix.tex Sec. C.3,
   "Effective mass formulas". Run from the repository root with:
   wolframscript -file "supplementary/reproduction/check_appendix_C3_effective_mass.wls"
*)

scriptDir = DirectoryName[$InputFileName];
If[scriptDir === "", scriptDir = Directory[]];
outFile = FileNameJoin[{scriptDir, "check_appendix_C3_effective_mass.out"}];

out = {};
results = {};
say[s_] := (Print[s]; AppendTo[out, ToString[s, InputForm]]);
record[name_, expr_] := Module[{ok = TrueQ[FullSimplify[expr, assumptions]]},
  AppendTo[results, {name, ok}];
  say[name <> ": " <> If[ok, "PASS", "FAIL"]];
  ok
];
recordWith[name_, expr_, extraAssumptions_] := Module[
  {ok = TrueQ[FullSimplify[expr, assumptions && extraAssumptions]]},
  AppendTo[results, {name, ok}];
  say[name <> ": " <> If[ok, "PASS", "FAIL"]];
  ok
];

assumptions =
  Element[{a, c, B, dmu, rhoAbs, Ddeg, k2}, Reals] &&
  a > 0 && c > 0 && B > 0 && rhoAbs > 0 && Ddeg > 0;

say["Appendix C.3 effective-mass Wolfram audit"];
say["Assumptions: a>0, c>0, B>0, rhoAbs=|rho_S|>0; dmu is real with branch restrictions when needed."];

vUpMinus = 2 a B dmu^2/(c^2 B - dmu^2);
vDownMinus = 2 a B dmu^2/(dmu^2 + c^2 B);
vUpPlus = 2 a B dmu^2/(dmu^2 - c^2 B);
vDownPlus = 2 a B dmu^2/(dmu^2 + c^2 B);

mUpMinus = (c^2 B - dmu^2)/(2 a dmu^2);
mDownMinus = 1/(2 a) + c^2 B/(2 a dmu^2);
mBulk = c^2 B/(2 a dmu^2);

recordWith[
  "Eq. C.28: m_eff(up-) = B/v_up-",
  B/vUpMinus == mUpMinus,
  dmu > 0 && dmu^2 < c^2 B
];

recordWith[
  "Eq. C.29: m_eff(down-) = B/v_down-",
  B/vDownMinus == mDownMinus,
  dmu < 0
];

recordWith[
  "Upper-branch parabolic mass limit, up+",
  Limit[B/vUpPlus, dmu -> Infinity] == 1/(2 a),
  dmu^2 > c^2 B
];

record[
  "Upper-branch parabolic mass limit, down+",
  Limit[B/vDownPlus, dmu -> Infinity] == 1/(2 a)
];

recordWith[
  "Eq. C.30 leading anomalous-bulk mass, up-",
  Series[mUpMinus, {dmu, 0, 2}] // Normal // (# == mBulk - 1/(2 a)) &,
  dmu > 0 && dmu^2 < c^2 B
];

recordWith[
  "Eq. C.30 leading anomalous-bulk mass, down-",
  Series[mDownMinus, {dmu, 0, 2}] // Normal // (# == mBulk + 1/(2 a)) &,
  dmu < 0
];

record[
  "Subleading constants are (-1/2a,+1/2a) for (up-,down-)",
  (mUpMinus - mBulk == -1/(2 a)) && (mDownMinus - mBulk == 1/(2 a))
];

DfromB = B/(2 Pi);
nstar = rhoAbs/DfromB;
record[
  "Eq. C.31: n*(B)=|rho_S|/D=2 Pi |rho_S|/B",
  nstar == 2 Pi rhoAbs/B
];

k2star = 2 nstar B;
record[
  "Eq. C.32: k_n*^2 ~= 2 n* B = 4 Pi |rho_S|",
  k2star == 4 Pi rhoAbs
];

Delta[k2_] := c^2/a + a k2;
uWeak = Delta[k2] + a B;
vWeak = Delta[k2] - a B;
ellUpMinusWeak = 1/2 (uWeak - Sqrt[uWeak^2 - 4 c^2 B]);
ellDownMinusWeak = 1/2 (vWeak - Sqrt[vWeak^2 + 4 c^2 B]);
recordWith[
  "Eq. C.33 input: up- weak-field dispersion has +c^2 B/Delta(k)",
  SeriesCoefficient[ellUpMinusWeak, {B, 0, 1}] == c^2/Delta[k2],
  k2 >= 0
];
recordWith[
  "Eq. C.33 input: down- weak-field dispersion has -c^2 B/Delta(k)",
  SeriesCoefficient[ellDownMinusWeak, {B, 0, 1}] == -c^2/Delta[k2],
  k2 >= 0
];

deltaMuFixedDensity[s_] := s c^2 B/Delta[k2star];
record[
  "Eq. C.33: fixed-density delta mu is linear in B",
  D[deltaMuFixedDensity[1], B] == c^2/Delta[4 Pi rhoAbs] &&
    D[deltaMuFixedDensity[-1], B] == -c^2/Delta[4 Pi rhoAbs]
];

mFixedDensity = FullSimplify[mBulk /. dmu -> deltaMuFixedDensity[1], assumptions];
mFixedDensityExpected = Delta[4 Pi rhoAbs]^2/(2 a c^2 B);
record[
  "Eq. C.34 first equality from substituting delta_mu(B)",
  mFixedDensity == mFixedDensityExpected
];

trMetric[k2_] := 2 c^2/Delta[k2]^2;
record[
  "Eq. C.34 second equality: Delta^2/(2ac^2 B)=1/(aB tr g)",
  mFixedDensityExpected == 1/(a B trMetric[4 Pi rhoAbs])
];

record[
  "Fixed-density scaling: m_eff(B,mu_rho(B)) is proportional to 1/B",
  FullSimplify[B D[Log[mFixedDensityExpected], B]] == -1
];

recordWith[
  "Fixed-mu scaling contrast: anomalous bulk mass is proportional to B at fixed dmu",
  FullSimplify[B D[Log[mBulk], B]] == 1,
  dmu != 0
];

record[
  "Absolute active-density convention keeps k_n*^2 positive",
  k2star > 0
];
say["NOTE: rhoAbs represents |rho_S| in Appendix C.3; the detailed sector convention is defined in Appendix D."];

failures = Select[results, ! TrueQ[#[[2]]] &];
say["Checks run: " <> ToString[Length[results]] <> "; failures: " <> ToString[Length[failures]]];
If[Length[failures] > 0, say["Failures: " <> ToString[failures, InputForm]]];

Export[outFile, StringRiffle[out, "\n"], "Text"];
If[Length[failures] > 0, Exit[1], Exit[0]];
