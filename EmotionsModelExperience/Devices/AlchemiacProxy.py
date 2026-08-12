
import sys
import asyncio
import struct
import csv
import numpy as np
from bleak import BleakClient, BleakScanner
from time import sleep
import numpy as np
import multiprocessing
#import sounddevice as sd
import platform
import asyncio
import datetime
from time import time

import threading

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque



ACC_SENS  = 0.061 / 1000      # 0.061 mg/LSB → g
GYRO_SENS  = 8.75 / 1000      # 8.75 mdps/LSB → dps
MAG_SENS  = 0.14 / 1000       # 0.14 mgauss/LSB → gauss


EEG_SERVICE_UUID = "9fa480e0-4967-11e5-a151-0002a5d5c51b"
EEG_DATA_UUID = "9fa480e1-4967-11e5-a151-0002a5d5c51b"
EEG_CONFIG_UUID = "9fa480e2-4967-11e5-a151-0002a5d5c51b"

EVENT_SERVICE_UUID = "9fa48300-4967-11e5-a151-0002a5d5c51b"
EVENT_UUID = "9fa48301-4967-11e5-a151-0002a5d5c51b"

MOTION_UUID = "9fa48201-4967-11e5-a151-0002a5d5c51b"
GYRO_UUID = "9fa48202-4967-11e5-a151-0002a5d5c51b"
COMPASS_UUID = "9fa48203-4967-11e5-a151-0002a5d5c51b"


SAMPLE_RATE = 250


          


class AlchemiacProxy:

    CONNECT_TIMEOUT = 30.0
    
    def __init__(self, mac_address, eeg_callback=None, motion_callback=None, event_callback=None, ):

        self.is_connected = False
        self.connection_error = None
        self.client = None
        self.last_packet = None
        self.packets = []
        self.samples_per_packets = []
        self.packet_received = []
        self.mac_address = mac_address
        
        self.eeg_queue = multiprocessing.Queue()
        self.motion_queue = multiprocessing.Queue()

        self.shutdown_event = asyncio.Event()
        self.loop = None  # Global to hold the event loop reference
        
        self.eeg_worker = threading.Thread(target=AlchemiacProxy.worker_process, args=(self.eeg_queue, eeg_callback))
        self.eeg_worker.start()

        self.motion_worker = threading.Thread(target=AlchemiacProxy.motion_process, args=(self.motion_queue, motion_callback))
        self.motion_worker.start()
        
        self.async_thread = threading.Thread(target=self.run_async_main)
        self.async_thread.start()
        
        pass
        
    def waitForConnected(self, timeout=None):
        if timeout is None:
            timeout = self.CONNECT_TIMEOUT
        deadline = time() + timeout
        while not self.is_connected:
            if self.connection_error:
                raise RuntimeError(
                    f"BLE connection failed for {self.mac_address}: {self.connection_error}"
                )
            if time() >= deadline:
                raise RuntimeError(
                    f"Timed out after {timeout:.0f}s waiting for BLE connection to "
                    f"{self.mac_address}. On macOS: power-cycle the headset, toggle "
                    "Bluetooth, and ensure no other app is connected."
                )
            sleep(0.1)
        
    def disconnect(self):
   
        try:
            self.trigger_shutdown()
            self.async_thread.join()
        finally:
            # Stop the worker process
            self.eeg_queue.put(None)  # Send sentinel to terminate the worker
            self.motion_queue.put(None)  # Send sentinel to terminate the worker
            self.eeg_worker.join()
            self.motion_worker.join()
        pass
        
        
    async def motion_handler(self, sender, data):
        try:
            now = datetime.datetime.now()
            ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw = struct.unpack_from('<hhhhhhhhh', data)
            timestamp_epoch = now.timestamp()
        
            self.motion_queue.put_nowait((timestamp_epoch, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw))
        except Exception as e:
            print(f"[AlchemiacProxy.py] motion_handler: queueing failed, somehow {e}")


        
    async def config_handler(self, sender, data):
        
        try:
            print(data)
        except Exception as e:
            print(f"[AlchemiacProxy.py] config_handler: failed {e}")


    async def notification_handler(self, sender, data):
        """Handler for receiving notifications from the button characteristic."""
        print(f"Notification from {sender}: {data}")



    async def packet_handler(self, client, sender, data):
        
        """Handler for receiving notifications from the new characteristic."""
        try:
            # Unpack the 192-byte packet into 8 samples, each with 8 channels (24 bits each)
            samples = []
            current_packet = data[0]
            # remove packet index, before parsing out the samples
            data = data[1:]
            
            missing_packets = self.detect_missing_packets(self.last_packet, current_packet)
            
            if len(missing_packets)>0:
                for packet in missing_packets:
                    self.packets.append(packet)
                    self.packet_received.append(False)
                    self.samples_per_packets.append(None)
            
            self.samples_per_packets.append(data)
            self.packet_received.append(True)
            self.packets.append(current_packet)
            self.last_packet = current_packet
            
            self.xfer_packets()

        except Exception as e:
            print(f"Error handling notification: {e}")
    

    def detect_missing_packets(self, last_packet, current_packet):
        missing_packets = []

        if last_packet is not None:
            if last_packet == 127:
                missing_packet = 0
                while missing_packet != current_packet:
                    print(f"Dropped packet {missing_packet}")
                    missing_packets.append(missing_packet)
                    missing_packet = (missing_packet + 1) % 128
            elif last_packet + 1 != current_packet:
                missing_packet = last_packet + 1
                while missing_packet != current_packet:
                    print(f"Dropped packet {missing_packet}")
                    missing_packets.append(missing_packet)
                    missing_packet = (missing_packet + 1) % 128

        return missing_packets

    def xfer_packets(self):

        while self.packet_received:
            delay = (self.last_packet + 128 - (self.packets[0] - 128)) % 128
            
            if not self.packet_received[0] and delay < 10:
                break

            # Enqueue the packet data for the worker process
            self.eeg_queue.put((self.packet_received[0],  self.samples_per_packets[0], self.packets[0]))

            # Remove the processed packet from the lists
            self.packet_received = self.packet_received[1:]
            self.samples_per_packets =  self.samples_per_packets[1:]
            self.packets = self.packets[1:]


    def run_async_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main_task(self.mac_address))
        
        
    def set_async_config(self):
        async def set_config_tmp(self):
            #await self.set_test_mode()
            await self.read_register()
        asyncio.run(set_config_tmp(self))   

    def trigger_shutdown(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.shutdown_event.set)

    
    async def main_task(self, deviceAddress, deviceName = "Hermes V1"):
        # 00:80:E1:27:47:BC

        self.client = BleakClient(deviceAddress)
        try:
            try:
                await self.client.connect(timeout=self.CONNECT_TIMEOUT)

                print(f"Connected to {deviceName} [{deviceAddress}]")
                self.is_connected = True
                start = time()

                # Subscribe to button notifications
                await self.client.start_notify(EVENT_UUID, self.notification_handler)
                print("Subscribed to button notifications.")

                # Subscribe to accelerometer notifications
                await self.client.start_notify(MOTION_UUID, self.motion_handler)
                print("Subscribed to accelerometer notifications.")

                # Subscribe to accelerometer notifications
                await self.client.start_notify(EEG_CONFIG_UUID, self.config_handler)
                print("Subscribed to accelerometer notifications.")

                # Subscribe to new characteristic notifications with asyncio.create_task
                await self.client.start_notify(
                    EEG_DATA_UUID,
                    lambda sender, data: asyncio.create_task(self.packet_handler(self.client, sender, data))
                )
                print("Subscribed to EEG data characteristic notifications.")

                await self.shutdown_event.wait()

                print(time()-start)

                await self.client.stop_notify(EEG_DATA_UUID)
                await self.client.stop_notify(EVENT_UUID)
                await self.client.stop_notify(MOTION_UUID)
                await self.client.stop_notify(EEG_CONFIG_UUID)
                print("Finished")
                self.is_connected = False
            except Exception as e:
                self.connection_error = str(e)
                print(f"[AlchemiacProxy] Connection failed: {e}")

        finally:
            if self.client and self.client.is_connected:
                await self.client.disconnect()
                print("Disconnected from device.")
            self.is_connected = False

    async def set_test_mode(self):
        if self.client and self.client.is_connected:
            config_bytes = bytes([0x03, 0x01, 0x00, 0x00])
            await self.client.write_gatt_char(EEG_CONFIG_UUID, config_bytes)
            print("Configuration bytes sent.")
        else:
            raise RuntimeError("Client is not connected.")
              
    
    async def read_register(self):
        if self.client and self.client.is_connected:
            config_bytes = bytes([0x02, 0x01, 0x01, 0x95])
            await self.client.write_gatt_char(EEG_CONFIG_UUID, config_bytes)
            print("Configuration bytes sent.")
            
            await asyncio.sleep(3)
            
            config_bytes = bytes([0x01, 0x01, 0x00, 0x00])
            await self.client.write_gatt_char(EEG_CONFIG_UUID, config_bytes)
            print("Start streaming sent.")
            
        else:
            raise RuntimeError("Client is not connected.")
              
              
    @staticmethod    
    def worker_process(eeg_queue, callback):
        """Worker process to write samples to a CSV file."""
        
        while True:
            try:
                # Get a task from the eeg_queue
                task = eeg_queue.get()
                if task is None:  # Sentinel to shut down the worker
                    break
                
                # Unpack the task
                packet_received, data, packet_number = task
                
                if packet_received:
                    samples = []
                    n_complete = len(data) // 24  # ignore any trailing partial chunk
                    if len(data) % 24 != 0:
                        print(f"Warning: packet {packet_number} has {len(data)} bytes "
                              f"(not a multiple of 24). Dropping {len(data) % 24} trailing byte(s).")
                    if n_complete == 0:
                        print(f"Warning: packet {packet_number} has no complete samples, skipping.")
                        continue
                    for i in range(0, n_complete * 24, 24):  # only complete 24-byte samples
                        sample = []
                        for j in range(0, 24, 3):  # 8 channels × 3 bytes each
                            channel_data = data[i + j:i + j + 3]
                            sample.append(int.from_bytes(channel_data, byteorder='big', signed=True))
                        samples.append(AlchemiacProxy.convert_ads1299_to_microvolts(sample))
                else:
                    # Fill with 10 x 8 NaN values
                    samples = [[np.nan] * 8 for _ in range(10)]
                
                # Write to the CSV file
                if callback is not None:
                    callback(samples)
                #packet_writer.writerow([packet_number, packet_received])
                
            except Exception as e:
                print(f"Error in worker process: {e}")
                break

    @staticmethod
    def motion_process(motion_queue, callback):
        filePath="data"

        while True:
            sample = motion_queue.get()
            
            if sample is None:  # Sentinel to shut down the writer
                break
            try:
                
                timestamp, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw = sample
                
                ax_raw *= ACC_SENS
                ay_raw *= ACC_SENS
                az_raw *= ACC_SENS
                
                gx_raw *= GYRO_SENS
                gy_raw *= GYRO_SENS
                gz_raw *= GYRO_SENS
                
                cx_raw *= MAG_SENS
                cy_raw *= MAG_SENS
                cz_raw *= MAG_SENS
                
                sample = (ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw)
                if callback is not None:
                    callback(sample)
            except Exception as e:
                print(f"[AlchemiacProxy.py] motion_process: Error: {e}")
              

    @staticmethod
    def convert_ads1299_to_microvolts(raw_values, gain=12, vref=4.5):
        """
        Converts raw 24-bit ADS1299 samples to microvolts.
        
        Args:
            raw_values: List or array of signed integers (24-bit).
            gain: Amplifier gain used (default 12).
            vref: Reference voltage (default 4.5V).

        Returns:
            List of floats in microvolts.
        """
        lsb_uV = (2 * vref * 1e6) / (gain * (2 ** 24))  # in microvolts
        return [val * lsb_uV for val in raw_values]

