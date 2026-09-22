import pyaudio

p = pyaudio.PyAudio()
print("\n--- DISPOSITIVI AUDIO DI INPUT DETECTED IN WSL ---")
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

input_devices_found = 0
for i in range(0, numdevices):
    device_info = p.get_device_info_by_host_api_device_index(0, i)
    if device_info.get('maxInputChannels') > 0:
        print(f"ID {i}: {device_info.get('name')} (Canali Input: {device_info.get('maxInputChannels')})")
        input_devices_found += 1

if input_devices_found == 0:
    print("ATTENZIONE: Nessun microfono rilevato da WSL!")
print("------------------------------------------------\n")
p.terminate()
