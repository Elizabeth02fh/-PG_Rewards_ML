# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['recompensas.py'],
             pathex=['F:\\9no SEMESTRE\\RECONOCIMIENTO DE PATRONES\\PROYECTO RP\\recompensas\\recompensas'],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.utils._weight_vector'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas+=[('favicon.ico','.\\favicon.ico','DATA'),
	('interfaz.ui','.\\interfaz.ui','DATA'),
	('modelo_entrenado_SVM.pkl','.\\modelo_entrenado_SVM.pkl','DATA'),
	]
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='recompensas',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
	  icon='favicon.ico' )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='recompensas')
