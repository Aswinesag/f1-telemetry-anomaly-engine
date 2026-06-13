import time
import json
import logging
from kafka import KafkaProducer, KafkaConsumer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TelemetryStreamProducer:
    """
    Simulates live car telemetry buses by streaming data rows sequentially 
    at a controlled temporal frequency.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "f1-telemetry-bus"):
        self.topic = topic
        # Initialize Kafka client with optimized serialization settings
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks=1, # Acknowledge message receipt at broker level for safety
            compression_type='gzip'
        )
        logging.info(f"Kafka Telemetry Producer bound to broker: {bootstrap_servers}")

    def stream_dataframe(self, dataframe, dynamic_hertz: int = 50):
        """
        Loops through a dataframe and publishes rows to Kafka to simulate live tracks.
        """
        delay_interval = 1.0 / dynamic_hertz
        logging.info(f"Initiating live streaming simulation at {dynamic_hertz}Hz...")
        
        for index, row in dataframe.iterrows():
            # Package row properties into a serializable payload dictionary
            message_payload = row.to_dict()
            message_payload['producer_timestamp'] = time.time()
            
            self.producer.send(self.topic, value=message_payload)
            time.sleep(delay_interval)
            
        self.producer.flush()
        logging.info("Historical data streaming playback completed.")


class TelemetryStreamConsumer:
    """
    Consumes live F1 streaming payloads from Kafka brokers to feed 
    downstream machine learning model inference heads.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "f1-telemetry-bus", group_id: str = "pitwall-analyser"):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='latest', # Focus strictly on real-time current telemetry frames
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logging.info(f"Kafka Telemetry Consumer subscribed to topic: {topic}")

    def start_inference_listening_loop(self, window_buffer_callback, sequence_size: int = 50):
        """
        Blocking event loop that accumulates live streaming samples into 
        a local FIFO rolling queue before executing model inference steps.
        """
        rolling_buffer = []
        logging.info("Inference engine consumer listening for active streams...")
        
        try:
            for message in self.consumer:
                telemetry_frame = message.value
                rolling_buffer.append(telemetry_frame)
                
                # Enforce strict FIFO bounding constraints to manage memory allocations
                if len(rolling_buffer) > sequence_size * 2:
                    rolling_buffer.pop(0)
                    
                # Once buffer satisfies minimal window boundaries, pass data to the models
                if len(rolling_buffer) >= sequence_size:
                    # Isolate exact window size required by the model architecture
                    active_inference_window = rolling_buffer[-sequence_size:]
                    window_buffer_callback(active_inference_window)
                    
        except KeyboardInterrupt:
            logging.info("Streaming consumption cleanly terminated by operator.")
        finally:
            self.consumer.close()