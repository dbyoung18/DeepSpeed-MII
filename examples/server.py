import mii
import socket

from request import Request, RequestGroup, RequestThread, RequestThreadPool
from streamer import TokenStreamer


class MiiServer:
    def __init__(self, args) -> None:
        self.deployment_name = args.deployment_name
        self.gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": False,  # Greedy
        }
        self.client = self.launch_serve(args)
        if args.stream:
            import nest_asyncio

            nest_asyncio.apply()

            self.streamer = TokenStreamer()
            self.pool = RequestThreadPool(1024)
        else:
            self.streamer = None
            self.pool = None

    def launch_serve(self, args) -> None:
        try:
            client = mii.serve(
                args.model,
                tensor_parallel=args.tp,
                replica_num=args.dp,
                deployment_name=args.deployment_name,
                port_number=args.port_number,
                torch_dist_port=args.torch_dist_port,
                zmq_port_number=args.zmq_port,
                skip_decode=args.skip_decode,
            )
            self.check_serve(args)
            return client
        except Exception as ex:
            print(f"EXCEPTION: {ex}")
            return None

    def check_serve(self, args):
        def generate_tp_ports(args):
            port_offset = 1
            tp_ports = []
            for i in range(args.dp):
                tensor_parallel = args.tp
                base_port = args.port_number + i * tensor_parallel + port_offset
                tp_ports += list(range(base_port, base_port + tensor_parallel))
            return tp_ports

        def check_socket(host, port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0

        tp_ports = generate_tp_ports(args)
        sockets_open = False
        while not sockets_open:
            sockets_open = check_socket("localhost", tp_ports[0])
        print(f"launched Mii Server {self.deployment_name}")

    def request(self, request: RequestGroup):
        responses = self.inference(request)
        return responses

    def inference(self, request: RequestGroup):
        responses = self.client.generate(
            request.prompts,
            request.input_tokens,
            uids=request.request_ids,
            max_new_tokens=self.gen_kwargs["max_new_tokens"],
            min_new_tokens=self.gen_kwargs["min_new_tokens"],
            do_sample=self.gen_kwargs["do_sample"],
        )
        return responses

    def async_request(self, request: Request):
        responses = self.pool.apply_async(self.async_inference, request)
        return responses

    def async_inference(self, request: Request):
        _ = self.client.generate(
            request.prompt,
            [request.input_tokens],
            uids=[request.request_id],
            max_new_tokens=self.gen_kwargs["max_new_tokens"],
            min_new_tokens=self.gen_kwargs["min_new_tokens"],
            do_sample=self.gen_kwargs["do_sample"],
            streaming_fn=self.streamer.put,
        )
        return self.streamer.generated_tokens

    def __del__(self):
        if self.client:
            self.client.terminate_server()
            print(f"terminated Mii Server {self.deployment_name}")
        if self.pool:
            self.pool.join()
