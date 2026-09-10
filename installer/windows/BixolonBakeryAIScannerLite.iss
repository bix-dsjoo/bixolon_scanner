#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef PayloadDir
  #error PayloadDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef VcRedistPath
  #error VcRedistPath is required
#endif
#ifndef SetupIconPath
  #error SetupIconPath is required
#endif

[Setup]
AppId={{95361E60-673F-4999-B5FD-89B6130447D5}
AppName=BIXOLON Bakery AI Scanner Lite
AppVersion={#AppVersion}
AppPublisher=BIXOLON
DefaultDirName={autopf}\BIXOLON Bakery AI Scanner Lite
DefaultGroupName=BIXOLON Bakery AI Scanner Lite
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=BixolonBakeryAIScannerLite-{#AppVersion}-Setup
SetupIconFile={#SetupIconPath}
UninstallDisplayIcon={app}\bakery_scanner_lite.exe
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
PrivilegesRequired=admin
CloseApplications=yes
RestartApplications=no
WizardStyle=modern
VersionInfoVersion={#AppVersion}
VersionInfoCompany=BIXOLON
VersionInfoDescription=BIXOLON Bakery AI Scanner Lite Installer
VersionInfoProductName=BIXOLON Bakery AI Scanner Lite
VersionInfoProductVersion={#AppVersion}
VersionInfoProductTextVersion={#AppVersion}
VersionInfoTextVersion={#AppVersion}

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#VcRedistPath}"; DestDir: "{tmp}"; DestName: "vc_redist.x64.exe"; Flags: deleteafterinstall

[Icons]
Name: "{autoprograms}\BIXOLON Bakery AI Scanner Lite"; Filename: "{app}\bakery_scanner_lite.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\BIXOLON Bakery AI Scanner Lite"; Filename: "{app}\bakery_scanner_lite.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Microsoft Visual C++ Runtime을 설치하는 중입니다..."; Flags: runhidden waituntilterminated
Filename: "{app}\bakery_scanner_lite.exe"; Description: "BIXOLON Bakery AI Scanner Lite 실행"; Flags: postinstall nowait skipifsilent unchecked
