#ifndef AppVersion
  #error AppVersion must be supplied by build_windows_installer.ps1
#endif
#ifndef PayloadDir
  #error PayloadDir must be supplied by build_windows_installer.ps1
#endif
#ifndef OutputDir
  #error OutputDir must be supplied by build_windows_installer.ps1
#endif
#ifndef SetupIconPath
  #error SetupIconPath must be supplied by build_windows_installer.ps1
#endif
#ifndef VcRedistPath
  #error VcRedistPath must be supplied by build_windows_installer.ps1
#endif

#define AppName "BIXOLON Bakery AI Scanner"
#define AppPublisher "BIXOLON"
#define AppExeName "product_scanner.exe"
#define PowerShellPath "{sys}\WindowsPowerShell\v1.0\powershell.exe"

[Setup]
AppId={{49706D67-B995-4B71-A49F-9F311D65165C}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\BIXOLON Bakery AI Scanner
DefaultGroupName=BIXOLON Bakery AI Scanner
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=BixolonBakeryAIScanner-{#AppVersion}-Setup
SetupIconFile={#SetupIconPath}
UninstallDisplayIcon={app}\{#AppExeName}
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
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Windows CPU Installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
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
Name: "{autoprograms}\BIXOLON Bakery AI Scanner"; Filename: "{#PowerShellPath}"; Parameters: "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\start-bixolon-scanner.ps1"""; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExeName}"; Comment: "BIXOLON Bakery AI Scanner CPU"
Name: "{autoprograms}\BIXOLON Bakery AI Scanner 설치 안내"; Filename: "{sys}\notepad.exe"; Parameters: """{app}\INSTALL-KO.txt"""; WorkingDir: "{app}"
Name: "{autodesktop}\BIXOLON Bakery AI Scanner"; Filename: "{#PowerShellPath}"; Parameters: "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\start-bixolon-scanner.ps1"""; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExeName}"; Comment: "BIXOLON Bakery AI Scanner CPU"; Tasks: desktopicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Microsoft Visual C++ Runtime을 설치하는 중입니다..."; Flags: runhidden waituntilterminated
Filename: "{#PowerShellPath}"; Parameters: "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\start-bixolon-scanner.ps1"""; Description: "BIXOLON Bakery AI Scanner 실행"; Flags: postinstall nowait skipifsilent unchecked
