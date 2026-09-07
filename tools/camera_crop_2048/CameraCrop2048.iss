#ifndef PayloadDir
  #error PayloadDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif

[Setup]
AppId={{AA85B293-FEEC-4710-AABC-72F7D1F3FB21}
AppName=Camera Crop 2048
AppVersion=1.0.0
DefaultDirName={localappdata}\Programs\Camera Crop 2048
DefaultGroupName=Camera Crop 2048
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=CameraCrop2048-1.0.0-Setup
UninstallDisplayIcon={app}\CameraCrop2048.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
PrivilegesRequired=lowest
CloseApplications=yes
RestartApplications=no
WizardStyle=modern
VersionInfoVersion=1.0.0.0
VersionInfoProductName=Camera Crop 2048
VersionInfoDescription=Camera Crop 2048 Installer
VersionInfoProductVersion=1.0.0
SetupLogging=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Camera Crop 2048"; Filename: "{app}\CameraCrop2048.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Camera Crop 2048"; Filename: "{app}\CameraCrop2048.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\CameraCrop2048.exe"; Description: "Camera Crop 2048 실행"; Flags: postinstall nowait skipifsilent
