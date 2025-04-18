belief network "unknown"
node AppOK {
  type : discrete [ 2 ] = { "Correct", "Incorrect_Corrupt" };
}
node DataFile {
  type : discrete [ 2 ] = { "Correct", "Incorrect_Corrupt" };
}
node AppData {
  type : discrete [ 2 ] = { "Correct", "Incorrect_or_corrupt" };
}
node DskLocal {
  type : discrete [ 2 ] = { "Greater_than_2_Mb", "Less_than_2_Mb" };
}
node PrtSpool {
  type : discrete [ 2 ] = { "Enabled", "Disabled" };
}
node PrtOn {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtPaper {
  type : discrete [ 2 ] = { "Has_Paper", "No_Paper" };
}
node NetPrint {
  type : discrete [ 2 ] = { "No__Local_printer_", "Yes__Network_printer_" };
}
node PrtDriver {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtThread {
  type : discrete [ 2 ] = { "OK", "Corrupt_Buggy" };
}
node EMFOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node GDIIN {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node DrvSet {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node DrvOK {
  type : discrete [ 2 ] = { "Reinstalled", "Corrupt" };
}
node GDIOUT {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtSel {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtDataOut {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtPath {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node NtwrkCnfg {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node PTROFFLINE {
  type : discrete [ 2 ] = { "Online", "Offline" };
}
node NetOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtCbl {
  type : discrete [ 2 ] = { "Connected", "Loose" };
}
node PrtPort {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node CblPrtHrdwrOK {
  type : discrete [ 2 ] = { "Operational", "Not_Operational" };
}
node LclOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node DSApplctn {
  type : discrete [ 2 ] = { "DOS", "Windows" };
}
node PrtMpTPth {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node DS_NTOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node DS_LCLOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PC2PRT {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtMem {
  type : discrete [ 2 ] = { "Greater_than_2_Mb", "Less_than_2Mb" };
}
node PrtTimeOut {
  type : discrete [ 2 ] = { "Long_Enough", "Too_Short" };
}
node FllCrrptdBffr {
  type : discrete [ 2 ] = { "Intact__not_Corrupt_", "Full_or_Corrupt" };
}
node TnrSpply {
  type : discrete [ 2 ] = { "Adequate", "Low" };
}
node PrtData {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node Problem1 {
  type : discrete [ 2 ] = { "Normal_Output", "No_Output" };
}
node AppDtGnTm {
  type : discrete [ 2 ] = { "Fast_Enough", "Too_Long" };
}
node PrntPrcssTm {
  type : discrete [ 2 ] = { "Fast_Enough", "Too_Long" };
}
node DeskPrntSpd {
  type : discrete [ 2 ] = { "OK", "Too_Slow" };
}
node PgOrnttnOK {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node PrntngArOK {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node ScrnFntNtPrntrFnt {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node CmpltPgPrntd {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node GrphcsRltdDrvrSttngs {
  type : discrete [ 2 ] = { "Correct", "Incorrect" };
}
node EPSGrphc {
  type : discrete [ 2 ] = { "No____TIF___WMF___BMP_", "Yes____EPS_" };
}
node NnPSGrphc {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtPScript {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PSGRAPHIC {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node Problem4 {
  type : discrete [ 2 ] = { "No", "Yes" };
}
node TrTypFnts {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node FntInstlltn {
  type : discrete [ 2 ] = { "Verified", "Faulty" };
}
node PrntrAccptsTrtyp {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node TTOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node NnTTOK {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node Problem5 {
  type : discrete [ 2 ] = { "No", "Yes" };
}
node LclGrbld {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node NtGrbld {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node GrbldOtpt {
  type : discrete [ 2 ] = { "No", "Yes" };
}
node HrglssDrtnAftrPrnt {
  type : discrete [ 2 ] = { "Fast_Enough", "Too_Long" };
}
node REPEAT {
  type : discrete [ 2 ] = { "Yes__Always_the_Same_", "No__Different_Each_Time_" };
}
node AvlblVrtlMmry {
  type : discrete [ 2 ] = { "Adequate____1Mb_", "Inadequate____1_Mb_" };
}
node PSERRMEM {
  type : discrete [ 2 ] = { "No_Error", "Low_Memory" };
}
node TstpsTxt {
  type : discrete [ 2 ] = { "x_1_Mb_Available_VM", "x_1_Mb_Available_VM2" };
}
node GrbldPS {
  type : discrete [ 2 ] = { "No", "Yes" };
}
node IncmpltPS {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtFile {
  type : discrete [ 2 ] = { "Yes", "No" };
}
node PrtIcon {
  type : discrete [ 2 ] = { "Normal", "Grayed_Out" };
}
node Problem6 {
  type : discrete [ 2 ] = { "No", "Yes" };
}
node Problem3 {
  type : discrete [ 2 ] = { "No", "Yes" };
}
node PrtQueue {
  type : discrete [ 2 ] = { "Short", "Long" };
}
node NtSpd {
  type : discrete [ 2 ] = { "OK", "Slow" };
}
node Problem2 {
  type : discrete [ 2 ] = { "OK", "Too_Long" };
}
node PrtStatPaper {
  type : discrete [ 2 ] = { "No_Error", "Jam__Out__Bin_Full" };
}
node PrtStatToner {
  type : discrete [ 2 ] = { "No_Error", "Low__None" };
}
node PrtStatMem {
  type : discrete [ 2 ] = { "No_Error", "Out_of_Memory" };
}
node PrtStatOff {
  type : discrete [ 2 ] = { "No_Error", "OFFLINE__OFF" };
}
probability ( AppOK ) {
   0.995, 0.005;
}
probability ( DataFile ) {
   0.995, 0.005;
}
probability ( AppData | AppOK, DataFile ) {
  (0, 0) : 0.9999, 0.0001;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.5, 0.5;
}
probability ( DskLocal ) {
   0.97, 0.03;
}
probability ( PrtSpool ) {
   0.95, 0.05;
}
probability ( PrtOn ) {
   0.9, 0.1;
}
probability ( PrtPaper ) {
   0.98, 0.02;
}
probability ( NetPrint ) {
   0.8, 0.2;
}
probability ( PrtDriver ) {
   0.9, 0.1;
}
probability ( PrtThread ) {
   0.9998, 0.0002;
}
probability ( EMFOK | AppData, DskLocal, PrtThread ) {
  (0, 0, 0) : 0.99, 0.01;
  (1, 0, 0) : 0.1, 0.9;
  (0, 1, 0) : 0.0, 1.0;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.05, 0.95;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( GDIIN | AppData, PrtSpool, EMFOK ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 1.0, 0.0;
  (1, 1, 0) : 0.0, 1.0;
  (0, 0, 1) : 0.0, 1.0;
  (1, 0, 1) : 0.0, 1.0;
  (0, 1, 1) : 1.0, 0.0;
  (1, 1, 1) : 0.0, 1.0;
}
probability ( DrvSet ) {
   0.99, 0.01;
}
probability ( DrvOK ) {
   0.99, 0.01;
}
probability ( GDIOUT | PrtDriver, GDIIN, DrvSet, DrvOK ) {
  (0, 0, 0, 0) : 0.99, 0.01;
  (1, 0, 0, 0) : 0.1, 0.9;
  (0, 1, 0, 0) : 0.1, 0.9;
  (1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0) : 0.9, 0.1;
  (1, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 1) : 0.2, 0.8;
  (1, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1) : 0.5, 0.5;
}
probability ( PrtSel ) {
   0.99, 0.01;
}
probability ( PrtDataOut | GDIOUT, PrtSel ) {
  (0, 0) : 0.99, 0.01;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.5, 0.5;
}
probability ( PrtPath ) {
   0.97, 0.03;
}
probability ( NtwrkCnfg ) {
   0.98, 0.02;
}
probability ( PTROFFLINE ) {
   0.7, 0.3;
}
probability ( NetOK | PrtPath, NtwrkCnfg, PTROFFLINE ) {
  (0, 0, 0) : 0.99, 0.01;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 0.1, 0.9;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.0, 1.0;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( PrtCbl ) {
   0.98, 0.02;
}
probability ( PrtPort ) {
   0.99, 0.01;
}
probability ( CblPrtHrdwrOK ) {
   0.99, 0.01;
}
probability ( LclOK | PrtCbl, PrtPort, CblPrtHrdwrOK ) {
  (0, 0, 0) : 0.999, 0.001;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 0.0, 1.0;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.01, 0.99;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( DSApplctn ) {
   0.15, 0.85;
}
probability ( PrtMpTPth ) {
   0.8, 0.2;
}
probability ( DS_NTOK | AppData, PrtPath, PrtMpTPth, NtwrkCnfg, PTROFFLINE ) {
  (0, 0, 0, 0, 0) : 0.99, 0.01;
  (1, 0, 0, 0, 0) : 0.2, 0.8;
  (0, 1, 0, 0, 0) : 0.0, 1.0;
  (1, 1, 0, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0, 0) : 0.0, 1.0;
  (1, 0, 1, 0, 0) : 0.5, 0.5;
  (0, 1, 1, 0, 0) : 0.5, 0.5;
  (1, 1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 0, 1, 0) : 0.1, 0.9;
  (1, 0, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 0, 1, 0) : 0.5, 0.5;
  (1, 1, 0, 1, 0) : 0.5, 0.5;
  (0, 0, 1, 1, 0) : 0.5, 0.5;
  (1, 0, 1, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 0, 1) : 0.0, 1.0;
  (1, 0, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 0, 1) : 0.5, 0.5;
  (1, 0, 1, 0, 1) : 0.5, 0.5;
  (0, 1, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 0, 1, 1) : 0.5, 0.5;
  (1, 1, 0, 1, 1) : 0.5, 0.5;
  (0, 0, 1, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1, 1) : 0.5, 0.5;
}
probability ( DS_LCLOK | AppData, PrtCbl, PrtPort, CblPrtHrdwrOK ) {
  (0, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 0, 0) : 0.1, 0.9;
  (0, 1, 0, 0) : 0.0, 1.0;
  (1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0) : 0.0, 1.0;
  (1, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 1) : 0.1, 0.9;
  (1, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1) : 0.5, 0.5;
}
probability ( PC2PRT | NetPrint, PrtDataOut, NetOK, LclOK, DSApplctn, DS_NTOK, DS_LCLOK ) {
  (0, 0, 0, 0, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 0, 0, 0, 0, 0) : 1.0, 0.0;
  (0, 1, 0, 0, 0, 0, 0) : 1.0, 0.0;
  (1, 1, 0, 0, 0, 0, 0) : 1.0, 0.0;
  (0, 0, 1, 0, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 1, 0, 0, 0, 0) : 1.0, 0.0;
  (0, 1, 1, 0, 0, 0, 0) : 1.0, 0.0;
  (1, 1, 1, 0, 0, 0, 0) : 1.0, 0.0;
  (0, 0, 0, 1, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 0, 1, 0, 0, 0) : 1.0, 0.0;
  (0, 1, 0, 1, 0, 0, 0) : 1.0, 0.0;
  (1, 1, 0, 1, 0, 0, 0) : 1.0, 0.0;
  (0, 0, 1, 1, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 1, 1, 0, 0, 0) : 1.0, 0.0;
  (0, 1, 1, 1, 0, 0, 0) : 1.0, 0.0;
  (1, 1, 1, 1, 0, 0, 0) : 1.0, 0.0;
  (0, 0, 0, 0, 1, 0, 0) : 1.0, 0.0;
  (1, 0, 0, 0, 1, 0, 0) : 1.0, 0.0;
  (0, 1, 0, 0, 1, 0, 0) : 0.0, 1.0;
  (1, 1, 0, 0, 1, 0, 0) : 0.0, 1.0;
  (0, 0, 1, 0, 1, 0, 0) : 1.0, 0.0;
  (1, 0, 1, 0, 1, 0, 0) : 0.0, 1.0;
  (0, 1, 1, 0, 1, 0, 0) : 0.0, 1.0;
  (1, 1, 1, 0, 1, 0, 0) : 0.0, 1.0;
  (0, 0, 0, 1, 1, 0, 0) : 0.0, 1.0;
  (1, 0, 0, 1, 1, 0, 0) : 1.0, 0.0;
  (0, 1, 0, 1, 1, 0, 0) : 0.0, 1.0;
  (1, 1, 0, 1, 1, 0, 0) : 0.0, 1.0;
  (0, 0, 1, 1, 1, 0, 0) : 0.0, 1.0;
  (1, 0, 1, 1, 1, 0, 0) : 0.0, 1.0;
  (0, 1, 1, 1, 1, 0, 0) : 0.0, 1.0;
  (1, 1, 1, 1, 1, 0, 0) : 0.0, 1.0;
  (0, 0, 0, 0, 0, 1, 0) : 1.0, 0.0;
  (1, 0, 0, 0, 0, 1, 0) : 0.0, 1.0;
  (0, 1, 0, 0, 0, 1, 0) : 1.0, 0.0;
  (1, 1, 0, 0, 0, 1, 0) : 0.0, 1.0;
  (0, 0, 1, 0, 0, 1, 0) : 1.0, 0.0;
  (1, 0, 1, 0, 0, 1, 0) : 0.0, 1.0;
  (0, 1, 1, 0, 0, 1, 0) : 1.0, 0.0;
  (1, 1, 1, 0, 0, 1, 0) : 0.0, 1.0;
  (0, 0, 0, 1, 0, 1, 0) : 1.0, 0.0;
  (1, 0, 0, 1, 0, 1, 0) : 0.0, 1.0;
  (0, 1, 0, 1, 0, 1, 0) : 1.0, 0.0;
  (1, 1, 0, 1, 0, 1, 0) : 0.0, 1.0;
  (0, 0, 1, 1, 0, 1, 0) : 1.0, 0.0;
  (1, 0, 1, 1, 0, 1, 0) : 0.0, 1.0;
  (0, 1, 1, 1, 0, 1, 0) : 1.0, 0.0;
  (1, 1, 1, 1, 0, 1, 0) : 0.0, 1.0;
  (0, 0, 0, 0, 1, 1, 0) : 1.0, 0.0;
  (1, 0, 0, 0, 1, 1, 0) : 1.0, 0.0;
  (0, 1, 0, 0, 1, 1, 0) : 0.0, 1.0;
  (1, 1, 0, 0, 1, 1, 0) : 0.0, 1.0;
  (0, 0, 1, 0, 1, 1, 0) : 1.0, 0.0;
  (1, 0, 1, 0, 1, 1, 0) : 0.0, 1.0;
  (0, 1, 1, 0, 1, 1, 0) : 0.0, 1.0;
  (1, 1, 1, 0, 1, 1, 0) : 0.0, 1.0;
  (0, 0, 0, 1, 1, 1, 0) : 0.0, 1.0;
  (1, 0, 0, 1, 1, 1, 0) : 1.0, 0.0;
  (0, 1, 0, 1, 1, 1, 0) : 0.0, 1.0;
  (1, 1, 0, 1, 1, 1, 0) : 0.0, 1.0;
  (0, 0, 1, 1, 1, 1, 0) : 0.0, 1.0;
  (1, 0, 1, 1, 1, 1, 0) : 0.0, 1.0;
  (0, 1, 1, 1, 1, 1, 0) : 0.0, 1.0;
  (1, 1, 1, 1, 1, 1, 0) : 0.0, 1.0;
  (0, 0, 0, 0, 0, 0, 1) : 0.0, 1.0;
  (1, 0, 0, 0, 0, 0, 1) : 1.0, 0.0;
  (0, 1, 0, 0, 0, 0, 1) : 0.0, 1.0;
  (1, 1, 0, 0, 0, 0, 1) : 1.0, 0.0;
  (0, 0, 1, 0, 0, 0, 1) : 0.0, 1.0;
  (1, 0, 1, 0, 0, 0, 1) : 1.0, 0.0;
  (0, 1, 1, 0, 0, 0, 1) : 0.0, 1.0;
  (1, 1, 1, 0, 0, 0, 1) : 1.0, 0.0;
  (0, 0, 0, 1, 0, 0, 1) : 0.0, 1.0;
  (1, 0, 0, 1, 0, 0, 1) : 1.0, 0.0;
  (0, 1, 0, 1, 0, 0, 1) : 0.0, 1.0;
  (1, 1, 0, 1, 0, 0, 1) : 1.0, 0.0;
  (0, 0, 1, 1, 0, 0, 1) : 0.0, 1.0;
  (1, 0, 1, 1, 0, 0, 1) : 1.0, 0.0;
  (0, 1, 1, 1, 0, 0, 1) : 0.0, 1.0;
  (1, 1, 1, 1, 0, 0, 1) : 1.0, 0.0;
  (0, 0, 0, 0, 1, 0, 1) : 1.0, 0.0;
  (1, 0, 0, 0, 1, 0, 1) : 1.0, 0.0;
  (0, 1, 0, 0, 1, 0, 1) : 0.0, 1.0;
  (1, 1, 0, 0, 1, 0, 1) : 0.0, 1.0;
  (0, 0, 1, 0, 1, 0, 1) : 1.0, 0.0;
  (1, 0, 1, 0, 1, 0, 1) : 0.0, 1.0;
  (0, 1, 1, 0, 1, 0, 1) : 0.0, 1.0;
  (1, 1, 1, 0, 1, 0, 1) : 0.0, 1.0;
  (0, 0, 0, 1, 1, 0, 1) : 0.0, 1.0;
  (1, 0, 0, 1, 1, 0, 1) : 1.0, 0.0;
  (0, 1, 0, 1, 1, 0, 1) : 0.0, 1.0;
  (1, 1, 0, 1, 1, 0, 1) : 0.0, 1.0;
  (0, 0, 1, 1, 1, 0, 1) : 0.0, 1.0;
  (1, 0, 1, 1, 1, 0, 1) : 0.0, 1.0;
  (0, 1, 1, 1, 1, 0, 1) : 0.0, 1.0;
  (1, 1, 1, 1, 1, 0, 1) : 0.0, 1.0;
  (0, 0, 0, 0, 0, 1, 1) : 0.0, 1.0;
  (1, 0, 0, 0, 0, 1, 1) : 0.0, 1.0;
  (0, 1, 0, 0, 0, 1, 1) : 0.0, 1.0;
  (1, 1, 0, 0, 0, 1, 1) : 0.0, 1.0;
  (0, 0, 1, 0, 0, 1, 1) : 0.0, 1.0;
  (1, 0, 1, 0, 0, 1, 1) : 0.0, 1.0;
  (0, 1, 1, 0, 0, 1, 1) : 0.0, 1.0;
  (1, 1, 1, 0, 0, 1, 1) : 0.0, 1.0;
  (0, 0, 0, 1, 0, 1, 1) : 0.0, 1.0;
  (1, 0, 0, 1, 0, 1, 1) : 0.0, 1.0;
  (0, 1, 0, 1, 0, 1, 1) : 0.0, 1.0;
  (1, 1, 0, 1, 0, 1, 1) : 0.0, 1.0;
  (0, 0, 1, 1, 0, 1, 1) : 0.0, 1.0;
  (1, 0, 1, 1, 0, 1, 1) : 0.0, 1.0;
  (0, 1, 1, 1, 0, 1, 1) : 0.0, 1.0;
  (1, 1, 1, 1, 0, 1, 1) : 0.0, 1.0;
  (0, 0, 0, 0, 1, 1, 1) : 1.0, 0.0;
  (1, 0, 0, 0, 1, 1, 1) : 1.0, 0.0;
  (0, 1, 0, 0, 1, 1, 1) : 0.0, 1.0;
  (1, 1, 0, 0, 1, 1, 1) : 0.0, 1.0;
  (0, 0, 1, 0, 1, 1, 1) : 1.0, 0.0;
  (1, 0, 1, 0, 1, 1, 1) : 0.0, 1.0;
  (0, 1, 1, 0, 1, 1, 1) : 0.0, 1.0;
  (1, 1, 1, 0, 1, 1, 1) : 0.0, 1.0;
  (0, 0, 0, 1, 1, 1, 1) : 0.0, 1.0;
  (1, 0, 0, 1, 1, 1, 1) : 1.0, 0.0;
  (0, 1, 0, 1, 1, 1, 1) : 0.0, 1.0;
  (1, 1, 0, 1, 1, 1, 1) : 0.0, 1.0;
  (0, 0, 1, 1, 1, 1, 1) : 0.0, 1.0;
  (1, 0, 1, 1, 1, 1, 1) : 0.0, 1.0;
  (0, 1, 1, 1, 1, 1, 1) : 0.0, 1.0;
  (1, 1, 1, 1, 1, 1, 1) : 0.0, 1.0;
}
probability ( PrtMem ) {
   0.95, 0.05;
}
probability ( PrtTimeOut ) {
   0.94, 0.06;
}
probability ( FllCrrptdBffr ) {
   0.85, 0.15;
}
probability ( TnrSpply ) {
   0.995, 0.005;
}
probability ( PrtData | PrtOn, PrtPaper, PC2PRT, PrtMem, PrtTimeOut, FllCrrptdBffr, TnrSpply ) {
  (0, 0, 0, 0, 0, 0, 0) : 0.99, 0.01;
  (1, 0, 0, 0, 0, 0, 0) : 0.0, 1.0;
  (0, 1, 0, 0, 0, 0, 0) : 0.0, 1.0;
  (1, 1, 0, 0, 0, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0, 0, 0, 0) : 0.0, 1.0;
  (1, 0, 1, 0, 0, 0, 0) : 0.5, 0.5;
  (0, 1, 1, 0, 0, 0, 0) : 0.5, 0.5;
  (1, 1, 1, 0, 0, 0, 0) : 0.5, 0.5;
  (0, 0, 0, 1, 0, 0, 0) : 0.1, 0.9;
  (1, 0, 0, 1, 0, 0, 0) : 0.5, 0.5;
  (0, 1, 0, 1, 0, 0, 0) : 0.5, 0.5;
  (1, 1, 0, 1, 0, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 1, 0, 0, 0) : 0.5, 0.5;
  (1, 0, 1, 1, 0, 0, 0) : 0.5, 0.5;
  (0, 1, 1, 1, 0, 0, 0) : 0.5, 0.5;
  (1, 1, 1, 1, 0, 0, 0) : 0.5, 0.5;
  (0, 0, 0, 0, 1, 0, 0) : 0.0, 1.0;
  (1, 0, 0, 0, 1, 0, 0) : 0.5, 0.5;
  (0, 1, 0, 0, 1, 0, 0) : 0.5, 0.5;
  (1, 1, 0, 0, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0, 1, 0, 0) : 0.5, 0.5;
  (1, 0, 1, 0, 1, 0, 0) : 0.5, 0.5;
  (0, 1, 1, 0, 1, 0, 0) : 0.5, 0.5;
  (1, 1, 1, 0, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 0, 1, 1, 0, 0) : 0.5, 0.5;
  (1, 0, 0, 1, 1, 0, 0) : 0.5, 0.5;
  (0, 1, 0, 1, 1, 0, 0) : 0.5, 0.5;
  (1, 1, 0, 1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 1, 1, 0, 0) : 0.5, 0.5;
  (1, 0, 1, 1, 1, 0, 0) : 0.5, 0.5;
  (0, 1, 1, 1, 1, 0, 0) : 0.5, 0.5;
  (1, 1, 1, 1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 0, 0, 0, 1, 0) : 0.02, 0.98;
  (1, 0, 0, 0, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 0, 0, 0, 1, 0) : 0.5, 0.5;
  (1, 1, 0, 0, 0, 1, 0) : 0.5, 0.5;
  (0, 0, 1, 0, 0, 1, 0) : 0.5, 0.5;
  (1, 0, 1, 0, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 0, 0, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 0, 0, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 1, 0, 1, 0) : 0.5, 0.5;
  (1, 0, 0, 1, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 0, 1, 0, 1, 0) : 0.5, 0.5;
  (1, 1, 0, 1, 0, 1, 0) : 0.5, 0.5;
  (0, 0, 1, 1, 0, 1, 0) : 0.5, 0.5;
  (1, 0, 1, 1, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 1, 0, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 1, 0, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 0, 1, 1, 0) : 0.5, 0.5;
  (1, 0, 0, 0, 1, 1, 0) : 0.5, 0.5;
  (0, 1, 0, 0, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 0, 0, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 1, 0, 1, 1, 0) : 0.5, 0.5;
  (1, 0, 1, 0, 1, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 0, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 0, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 1, 1, 1, 0) : 0.5, 0.5;
  (1, 0, 0, 1, 1, 1, 0) : 0.5, 0.5;
  (0, 1, 0, 1, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 0, 1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 1, 1, 1, 1, 0) : 0.5, 0.5;
  (1, 0, 1, 1, 1, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 1, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 0, 0, 0, 1) : 0.01, 0.99;
  (1, 0, 0, 0, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 0, 0, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 0, 0, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 0, 0, 0, 1) : 0.5, 0.5;
  (1, 0, 1, 0, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 1, 0, 0, 0, 1) : 0.5, 0.5;
  (1, 1, 1, 0, 0, 0, 1) : 0.5, 0.5;
  (0, 0, 0, 1, 0, 0, 1) : 0.5, 0.5;
  (1, 0, 0, 1, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 1, 0, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 1, 0, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 1, 0, 0, 1) : 0.5, 0.5;
  (1, 0, 1, 1, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 1, 1, 0, 0, 1) : 0.5, 0.5;
  (1, 1, 1, 1, 0, 0, 1) : 0.5, 0.5;
  (0, 0, 0, 0, 1, 0, 1) : 0.5, 0.5;
  (1, 0, 0, 0, 1, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 0, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 0, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 0, 1, 0, 1) : 0.5, 0.5;
  (1, 0, 1, 0, 1, 0, 1) : 0.5, 0.5;
  (0, 1, 1, 0, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 1, 0, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 0, 1, 1, 0, 1) : 0.5, 0.5;
  (1, 0, 0, 1, 1, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 1, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 1, 1, 0, 1) : 0.5, 0.5;
  (1, 0, 1, 1, 1, 0, 1) : 0.5, 0.5;
  (0, 1, 1, 1, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 1, 1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 0, 0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 0, 0, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 0, 0, 0, 1, 1) : 0.5, 0.5;
  (1, 1, 0, 0, 0, 1, 1) : 0.5, 0.5;
  (0, 0, 1, 0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 0, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 0, 0, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 0, 0, 1, 1) : 0.5, 0.5;
  (0, 0, 0, 1, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 0, 1, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 0, 1, 0, 1, 1) : 0.5, 0.5;
  (1, 1, 0, 1, 0, 1, 1) : 0.5, 0.5;
  (0, 0, 1, 1, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1, 0, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1, 0, 1, 1) : 0.5, 0.5;
  (0, 0, 0, 0, 1, 1, 1) : 0.5, 0.5;
  (1, 0, 0, 0, 1, 1, 1) : 0.5, 0.5;
  (0, 1, 0, 0, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 0, 0, 1, 1, 1) : 0.5, 0.5;
  (0, 0, 1, 0, 1, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 0, 1, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 0, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 0, 1, 1, 1) : 0.5, 0.5;
  (0, 0, 0, 1, 1, 1, 1) : 0.5, 0.5;
  (1, 0, 0, 1, 1, 1, 1) : 0.5, 0.5;
  (0, 1, 0, 1, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 0, 1, 1, 1, 1) : 0.5, 0.5;
  (0, 0, 1, 1, 1, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1, 1, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1, 1, 1, 1) : 0.5, 0.5;
}
probability ( Problem1 | PrtData ) {
  (0) : 1.0, 0.0;
  (1) : 0.0, 1.0;
}
probability ( AppDtGnTm | PrtSpool ) {
  (0) : 0.998, 0.002;
  (1) : 0.99000001, 0.00999999;
}
probability ( PrntPrcssTm | PrtSpool ) {
  (0) : 0.99000001, 0.00999999;
  (1) : 1.0, 0.0;
}
probability ( DeskPrntSpd | PrtMem, AppDtGnTm, PrntPrcssTm ) {
  (0, 0, 0) : 0.99900001, 0.00099999;
  (1, 0, 0) : 0.25, 0.75;
  (0, 1, 0) : 0.00099999, 0.99900001;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.00099999, 0.99900001;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( PgOrnttnOK ) {
   0.95, 0.05;
}
probability ( PrntngArOK ) {
   0.98, 0.02;
}
probability ( ScrnFntNtPrntrFnt ) {
   0.95, 0.05;
}
probability ( CmpltPgPrntd | PrtMem, PgOrnttnOK, PrntngArOK ) {
  (0, 0, 0) : 0.99, 0.01;
  (1, 0, 0) : 0.3, 0.7;
  (0, 1, 0) : 0.00999999, 0.99000001;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.1, 0.9;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( GrphcsRltdDrvrSttngs ) {
   0.95, 0.05;
}
probability ( EPSGrphc ) {
   0.99, 0.01;
}
probability ( NnPSGrphc | PrtMem, GrphcsRltdDrvrSttngs, EPSGrphc ) {
  (0, 0, 0) : 0.999, 0.001;
  (1, 0, 0) : 0.25, 0.75;
  (0, 1, 0) : 0.1, 0.9;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.0, 1.0;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( PrtPScript ) {
   0.4, 0.6;
}
probability ( PSGRAPHIC | PrtMem, GrphcsRltdDrvrSttngs, EPSGrphc ) {
  (0, 0, 0) : 0.999, 0.001;
  (1, 0, 0) : 0.25, 0.75;
  (0, 1, 0) : 0.1, 0.9;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( Problem4 | NnPSGrphc, PrtPScript, PSGRAPHIC ) {
  (0, 0, 0) : 0.0, 1.0;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 0.0, 1.0;
  (1, 1, 0) : 1.0, 0.0;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 1.0, 0.0;
  (0, 1, 1) : 0.0, 1.0;
  (1, 1, 1) : 1.0, 0.0;
}
probability ( TrTypFnts ) {
   0.9, 0.1;
}
probability ( FntInstlltn ) {
   0.98, 0.02;
}
probability ( PrntrAccptsTrtyp ) {
   0.9, 0.1;
}
probability ( TTOK | PrtMem, FntInstlltn, PrntrAccptsTrtyp ) {
  (0, 0, 0) : 0.99000001, 0.00999999;
  (1, 0, 0) : 0.5, 0.5;
  (0, 1, 0) : 0.1, 0.9;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.0, 1.0;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( NnTTOK | PrtMem, ScrnFntNtPrntrFnt, FntInstlltn ) {
  (0, 0, 0) : 0.99000001, 0.00999999;
  (1, 0, 0) : 0.5, 0.5;
  (0, 1, 0) : 0.1, 0.9;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.1, 0.9;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( Problem5 | TrTypFnts, TTOK, NnTTOK ) {
  (0, 0, 0) : 0.0, 1.0;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 1.0, 0.0;
  (1, 1, 0) : 0.0, 1.0;
  (0, 0, 1) : 0.0, 1.0;
  (1, 0, 1) : 1.0, 0.0;
  (0, 1, 1) : 1.0, 0.0;
  (1, 1, 1) : 1.0, 0.0;
}
probability ( LclGrbld | AppData, PrtDriver, PrtMem, CblPrtHrdwrOK ) {
  (0, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 0, 0) : 0.2, 0.8;
  (0, 1, 0, 0) : 0.4, 0.6;
  (1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0) : 0.2, 0.8;
  (1, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 1) : 0.1, 0.9;
  (1, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1) : 0.5, 0.5;
}
probability ( NtGrbld | AppData, PrtDriver, PrtMem, NtwrkCnfg ) {
  (0, 0, 0, 0) : 1.0, 0.0;
  (1, 0, 0, 0) : 0.3, 0.7;
  (0, 1, 0, 0) : 0.4, 0.6;
  (1, 1, 0, 0) : 0.5, 0.5;
  (0, 0, 1, 0) : 0.2, 0.8;
  (1, 0, 1, 0) : 0.5, 0.5;
  (0, 1, 1, 0) : 0.5, 0.5;
  (1, 1, 1, 0) : 0.5, 0.5;
  (0, 0, 0, 1) : 0.4, 0.6;
  (1, 0, 0, 1) : 0.5, 0.5;
  (0, 1, 0, 1) : 0.5, 0.5;
  (1, 1, 0, 1) : 0.5, 0.5;
  (0, 0, 1, 1) : 0.5, 0.5;
  (1, 0, 1, 1) : 0.5, 0.5;
  (0, 1, 1, 1) : 0.5, 0.5;
  (1, 1, 1, 1) : 0.5, 0.5;
}
probability ( GrbldOtpt | NetPrint, LclGrbld, NtGrbld ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 1.0, 0.0;
  (0, 1, 0) : 0.0, 1.0;
  (1, 1, 0) : 1.0, 0.0;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 0.0, 1.0;
  (0, 1, 1) : 0.0, 1.0;
  (1, 1, 1) : 0.0, 1.0;
}
probability ( HrglssDrtnAftrPrnt | AppDtGnTm ) {
  (0) : 0.99, 0.01;
  (1) : 0.1, 0.9;
}
probability ( REPEAT | CblPrtHrdwrOK, NtwrkCnfg ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.5, 0.5;
  (0, 1) : 0.5, 0.5;
  (1, 1) : 0.5, 0.5;
}
probability ( AvlblVrtlMmry | PrtPScript ) {
  (0) : 0.98, 0.02;
  (1) : 1.0, 0.0;
}
probability ( PSERRMEM | PrtPScript, AvlblVrtlMmry ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 1.0, 0.0;
  (0, 1) : 0.05, 0.95;
  (1, 1) : 1.0, 0.0;
}
probability ( TstpsTxt | PrtPScript, AvlblVrtlMmry ) {
  (0, 0) : 0.99900001, 0.00099999;
  (1, 0) : 1.0, 0.0;
  (0, 1) : 0.00099999, 0.99900001;
  (1, 1) : 1.0, 0.0;
}
probability ( GrbldPS | GrbldOtpt, AvlblVrtlMmry ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.1, 0.9;
  (1, 1) : 0.5, 0.5;
}
probability ( IncmpltPS | CmpltPgPrntd, AvlblVrtlMmry ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.3, 0.7;
  (1, 1) : 0.5, 0.5;
}
probability ( PrtFile | PrtDataOut ) {
  (0) : 0.8, 0.2;
  (1) : 0.2, 0.8;
}
probability ( PrtIcon | NtwrkCnfg, PTROFFLINE ) {
  (0, 0) : 0.9999, 0.0001;
  (1, 0) : 0.25, 0.75;
  (0, 1) : 0.7, 0.3;
  (1, 1) : 0.5, 0.5;
}
probability ( Problem6 | GrbldOtpt, PrtPScript, GrbldPS ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 1.0, 0.0;
  (0, 1, 0) : 1.0, 0.0;
  (1, 1, 0) : 0.0, 1.0;
  (0, 0, 1) : 0.0, 1.0;
  (1, 0, 1) : 0.0, 1.0;
  (0, 1, 1) : 1.0, 0.0;
  (1, 1, 1) : 0.0, 1.0;
}
probability ( Problem3 | CmpltPgPrntd, PrtPScript, IncmpltPS ) {
  (0, 0, 0) : 0.0, 1.0;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 0.0, 1.0;
  (1, 1, 0) : 1.0, 0.0;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 1.0, 0.0;
  (0, 1, 1) : 0.0, 1.0;
  (1, 1, 1) : 1.0, 0.0;
}
probability ( PrtQueue ) {
   0.99, 0.01;
}
probability ( NtSpd | DeskPrntSpd, NtwrkCnfg, PrtQueue ) {
  (0, 0, 0) : 0.99900001, 0.00099999;
  (1, 0, 0) : 0.0, 1.0;
  (0, 1, 0) : 0.25, 0.75;
  (1, 1, 0) : 0.5, 0.5;
  (0, 0, 1) : 0.25, 0.75;
  (1, 0, 1) : 0.5, 0.5;
  (0, 1, 1) : 0.5, 0.5;
  (1, 1, 1) : 0.5, 0.5;
}
probability ( Problem2 | NetPrint, DeskPrntSpd, NtSpd ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 1.0, 0.0;
  (0, 1, 0) : 0.0, 1.0;
  (1, 1, 0) : 1.0, 0.0;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 0.0, 1.0;
  (0, 1, 1) : 0.0, 1.0;
  (1, 1, 1) : 0.0, 1.0;
}
probability ( PrtStatPaper | PrtPaper ) {
  (0) : 0.99900001, 0.00099999;
  (1) : 0.00099999, 0.99900001;
}
probability ( PrtStatToner | TnrSpply ) {
  (0) : 0.99900001, 0.00099999;
  (1) : 0.00099999, 0.99900001;
}
probability ( PrtStatMem | PrtMem ) {
  (0) : 0.99900001, 0.00099999;
  (1) : 0.2, 0.8;
}
probability ( PrtStatOff | PrtOn ) {
  (0) : 0.99000001, 0.00999999;
  (1) : 0.00999999, 0.99000001;
}
