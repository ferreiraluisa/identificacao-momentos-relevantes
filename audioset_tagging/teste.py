import h5py

with h5py.File('hdf5s/waveforms/eval.h5', 'r') as arquivo:
    print("Grupos disponíveis:", list(arquivo.keys()))

    grupo = arquivo['audio_name']
    audio_names = [name.decode('utf-8') for name in grupo[:]]
    
    print("Nomes dos áudios:", audio_names)

    targets = arquivo['target'][:]
    print(targets) 
    
    waveforms = arquivo['waveform'][:]
    print("Waveforms shape:", waveforms.shape)
