import torch
from torch import tensor
print(torch.round(tensor(6)/tensor(5)))



class SAR_ADC:
    def __init__(self, resolution):
        self.resolution = resolution  # Resolution of the ADC
        self.max_value = (1 << resolution) - 1  # Maximum digital output value
        self.bits = [0] * resolution  # Initialize SAR bits
        
    def convert(self, analog_input):
        # Perform SAR conversion
        for i in range(self.resolution):
            # Set SAR bit to 1
            self.bits[i] = 1
            # Check if analog input is greater than threshold
            if self.analog_output() > analog_input:
                self.bits[i] = 0  # Set SAR bit to 0 if analog input is greater
        return self.analog_output()
    
    def analog_output(self):
        # Convert SAR bits to analog output value
        return sum(self.bits[i] * (1 << (self.resolution - i - 1)) for i in range(self.resolution))
    

# Example usage:
resolution = 8  # 8-bit resolution ADC
sar_adc = SAR_ADC(resolution)

# Perform ADC conversion
analog_input = 0.75  # Analog input voltage (0 to 1)
digital_output = sar_adc.convert(analog_input)
print(f"Analog Input: {analog_input}, Digital Output: {digital_output}, Binary Output: {bin(digital_output)}")
