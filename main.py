import json
import math
import random
import select
import socket

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty, ObjectProperty, NumericProperty, ColorProperty, BooleanProperty, \
    StringProperty
from kivy.clock import Clock
from kivy.core.window import Window

class RubberObstacle(Widget):
    middle_pos = NumericProperty(0)
    #
    # def get_color(self):
    #     c=ColorProperty(1, 1, 1)
    #     print(c)
    #     return c

class TimeGauge(Widget):
    green_height_percentage = NumericProperty(0)

class Player(Widget):
    pass

class Target(Widget):
    pass

class CrossHair(Widget):
    pass

class ConnectionIndicator(Label):
    connected = BooleanProperty(False)

class SliderInterfaceRoot(BoxLayout):

    player = ObjectProperty(None)
    target = ObjectProperty(None)
    gauge = ObjectProperty(None)
    rubber = ObjectProperty(None)
    conn_indicator = ObjectProperty(None)
    server_ip = ObjectProperty(None)
    server_port = ObjectProperty(None)
    connect_button = ObjectProperty(None)
    status_message = StringProperty("")

    position_error = NumericProperty(0)

    position_tolerance = NumericProperty(5)

    green_increase = 0.02
    green_decrease = 0.04


    UDPClientSocket = None
    bufferSize= 1024
    connection_failures = 0

    circularBufferSize = 100

    current_time = 0

    is_connected = False

    section = "left" # or "right"

    def __init__(self, **kwargs):
        super(SliderInterfaceRoot, self).__init__(**kwargs)
        self.player_on_target_sentinel = None
        self.target_pos_limit = 900
        self.position_tolerance = (self.target.size[0] * self.target.out_scale - self.player.size[0])/2
        Clock.schedule_once(self.update_target_pos, 4)

    def on_start_of_target(self, dt):
        if self.position_error < self.position_tolerance:
            self.gauge.green_height_percentage = self.gauge.green_height_percentage + self.green_increase
            if self.gauge.green_height_percentage >= 1:
                Clock.unschedule(self.player_on_target_sentinel)
                self.player_on_target_sentinel = None
                self.update_target_pos()
                self.gauge.green_height_percentage = 0
        else:
            self.gauge.green_height_percentage = self.gauge.green_height_percentage - self.green_decrease
            if self.gauge.green_height_percentage <= 0:
                Clock.unschedule(self.player_on_target_sentinel)
                self.player_on_target_sentinel = None
                self.gauge.green_height_percentage = 0

    def update_target_pos(self, *args, **kwargs):
        if not self.get_root_window() is None:
            if self.section == "left":
                self.target.pos[0] = self.get_root_window().size[0]/4 + (0-.5)*2*self.get_root_window().size[0]/4/3
                self.section = "right"
            else:
                self.target.pos[0] = self.get_root_window().size[0]*5/8 + (1-.5)*2*self.get_root_window().size[0]/4/3
                self.section = "left"

            # self.target.pos[0] = (random.Random().random()-.5)*2*self.target_pos_limit + self.size[0]/2
            self.rubber.pos[0] = (self.target.pos[0] + self.player.pos[0])/2

            self.rubber.middle_pos = 0

            self.from_player_to_rubber = self.rubber.pos[0] - self.player.pos[0]
            self.from_player_to_target = self.target.pos[0] - self.player.pos[0]
            if self.from_player_to_target - self.from_player_to_rubber == 0:
                self.update_target_pos()
                return
            self.target_way = (self.from_player_to_target - self.from_player_to_rubber) / abs(self.from_player_to_target - self.from_player_to_rubber)

    def on_touch_move(self, touch):
        self.player.pos[0] = touch.x
        self.update_rubber_displacement()

    def update_rubber_displacement(self):
        self.from_player_to_rubber = self.rubber.pos[0] - self.player.pos[0]
        self.rubber.middle_pos = -min(self.from_player_to_rubber, 0) if self.target_way > 0 else -max(self.from_player_to_rubber, 0)
        self.position_error = abs(self.target.pos[0] - self.player.pos[0])
        if self.position_error < self.position_tolerance and self.player_on_target_sentinel == None:
            self.player_on_target_sentinel = Clock.schedule_interval(self.on_start_of_target, 0.05)

    def setup_connection(self, button):
        if not self.is_connected:
            handShakeFromClient = "hello passivity-project-server"
            bytesToSend = str.encode(handShakeFromClient)

            # Create a UDP socket at client side
            self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self.UDPClientSocket.settimeout(0) # non blocking!

            inputs = [self.UDPClientSocket]
            outputs = []
            timeout = 0.5

            # Send to server using created UDP socket
            self.UDPClientSocket.sendto(bytesToSend, (self.server_ip.text, int(self.server_port.text)))

            # self.ids.connection_status_text = "Trying connection"
            readable, writable, exceptional = select.select(inputs, outputs, inputs, timeout)

            if self.UDPClientSocket in readable:
                msgFromServer = self.UDPClientSocket.recvfrom(self.bufferSize)

                msgJson = json.loads(msgFromServer[0])

                try:
                    if msgJson['comment'] == 'hi simple-slider-game':

                        self.connection_failures = 0
                        self.status_message = "Connection success!"
                        self.try_connection_btn_text = "Disconnect"
                        self.server_ip.disabled = True
                        self.server_port.disabled = True

                        self.conn_indicator.connected = True
                        self.is_connected = True

                        self.game_update_event = Clock.schedule_interval(self.game_update, 0.01)
                except:
                    self.connection_failures += 1
                    self.status_message = "Couldn't connect ({})".format(self.connection_failures)
            else:
                self.connection_failures += 1
                self.status_message = "Couldn't connect ({})".format(self.connection_failures)
        else:
            bytesToSend = str.encode('end UDP communication')
            self.UDPClientSocket.sendto(bytesToSend, (self.server_ip.text, int(self.server_port.text)))
            print(bytesToSend, (self.server_ip.text, int(self.server_port.text)))
            self.is_connected = False
            self.conn_indicator.connected = False

    def game_update(self, dt):
        try:
            msgFromServer = self.UDPClientSocket.recvfrom(self.bufferSize)
        except:
            return
        msgJson = json.loads(msgFromServer[0])
        if msgJson['Type'] == 'epos_info':
            # self.player.pos[0] = (-float(msgJson['position'])+math.pi/2)*self.get_root_window().size[0]/(math.pi)
            self.player.pos[0] = self.get_root_window().size[0]/2 +(-float(msgJson['position']))*(self.get_root_window().size[0]/4)/(math.pi/12)
            self.update_rubber_displacement()



class SliderInterfaceApp(App):
    def build(self):
        s = SliderInterfaceRoot()
        s.update_target_pos()
        Window.maximize()
        return s

SliderInterfaceApp().run()