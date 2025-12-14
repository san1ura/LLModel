"""
Model serving and inference for transformer models
Supports multiple serving options and optimized inference
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any, Union
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np


class InferenceEngine:
    """
    High-performance inference engine for transformer models
    """
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare device
        self.device = config.device if hasattr(config, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Initialize KV cache manager if supported
        self.kv_cache_manager = None
        if hasattr(config, 'use_kv_cache') and config.use_kv_cache:
            from model.layers.kv_cache import KVCacheManager
            self.kv_cache_manager = KVCacheManager(
                max_batch_size=1,  # Default for single generation
                max_seq_len=config.max_len,
                num_heads=config.n_heads,
                head_dim=config.d_model // config.n_heads
            )
        
        # Initialize generation parameters
        self.default_params = {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.2,
            'max_new_tokens': 100,
            'do_sample': True
        }
    
    def preprocess_inputs(self, inputs: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Preprocess input texts to model inputs"""
        if isinstance(inputs, str):
            inputs = [inputs]

        # Tokenize inputs
        # Since we're using our custom tokenizer, we need to use its interface
        input_ids_list = [torch.tensor(self.tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)
                         for text in inputs]

        # Find max length
        max_len = max(len(ids) for ids in input_ids_list)

        # Pad sequences to the same length
        padded_input_ids = []
        for ids in input_ids_list:
            ids_tensor = torch.tensor(ids, dtype=torch.long) if not torch.is_tensor(ids) else ids.clone().detach()
            # Pad if necessary
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', self.config.pad_token_id)
            if callable(pad_token_id):
                pad_token_id = pad_token_id()
            if len(ids_tensor) < max_len:
                padded_ids = torch.nn.functional.pad(ids_tensor, (0, max_len - len(ids_tensor)), value=int(pad_token_id))
            else:
                padded_ids = ids_tensor

            padded_input_ids.append(padded_ids)

        input_ids = torch.stack(padded_input_ids).to(self.device)

        # Create attention mask (1 for real tokens, 0 for padding) - but we won't pass this to the model for now
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', self.config.pad_token_id)
        if callable(pad_token_id):
            pad_token_id = pad_token_id()
        attention_mask = (input_ids != int(pad_token_id)).long().to(self.device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask  # Still return for compatibility, but won't use in model forward call
        }
    
    def postprocess_outputs(self, output_ids: torch.Tensor) -> List[str]:
        """Postprocess model outputs to strings"""
        # Decode token IDs to text
        output_list = []
        for seq in output_ids:
            decoded = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            output_list.append(decoded)

        return output_list
    
    def generate(
        self,
        inputs: Union[str, List[str]],
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        num_beams: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using the model
        """
        # If it's a single string, convert to list for processing
        if isinstance(inputs, str):
            inputs = [inputs]

        # Tokenize inputs
        input_ids_list = [torch.tensor(self.tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)
                         for text in inputs]

        # Pad sequences to same length
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        for ids in input_ids_list:
            ids_tensor = torch.tensor(ids, dtype=torch.long) if not torch.is_tensor(ids) else ids.clone().detach()
            if len(ids_tensor) < max_len:
                padded_ids = torch.nn.functional.pad(ids_tensor, (0, max_len - len(ids_tensor)), value=0)
            else:
                padded_ids = ids_tensor
            padded_input_ids.append(padded_ids)

        input_ids = torch.stack(padded_input_ids).to(self.device)

        # Use the model's built-in generate method which should handle everything properly
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', self.config.pad_token_id)
            if callable(pad_token_id):
                pad_token_id = pad_token_id()
        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', self.config.eos_token_id)
            if callable(eos_token_id):
                eos_token_id = eos_token_id()

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty
            )

        # Decode the generated sequences
        results = []
        for seq in generated:
            decoded = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            results.append(decoded)

        # Return single string if input was single string
        if len(results) == 1 and isinstance(inputs, list) and len(inputs) == 1:
            return results[0]
        else:
            return results
    
    def generate_stream(
        self,
        inputs: str,
        **generation_kwargs
    ):
        """
        Stream tokens one by one for real-time generation
        """
        # This is a simplified version that yields tokens after full generation
        # A true streaming implementation would yield tokens as they're generated
        result = self.generate(inputs, **generation_kwargs)
        # Yield tokens one by one
        tokens = self.tokenizer.encode(result)
        for token_id in tokens:
            yield self.tokenizer.decode([token_id])
    
    def batch_generate(
        self,
        inputs: List[str],
        **generation_kwargs
    ) -> List[str]:
        """
        Generate for a batch of inputs
        """
        results = []
        for input_text in inputs:
            result = self.generate(input_text, **generation_kwargs)
            results.append(result)
        return results
    
    def benchmark_inference(
        self,
        input_text: str = "Once upon a time",
        num_generations: int = 10,
        **generation_kwargs
    ) -> Dict[str, float]:
        """
        Benchmark inference performance
        """
        times = []
        lengths = []
        
        for _ in range(num_generations):
            start_time = time.time()
            result = self.generate(input_text, **generation_kwargs)
            end_time = time.time()
            
            times.append(end_time - start_time)
            lengths.append(len(result.split()))
        
        avg_time = sum(times) / len(times)
        avg_length = sum(lengths) / len(lengths)
        throughput = avg_length / avg_time  # tokens per second
        
        return {
            'avg_generation_time': avg_time,
            'avg_num_tokens': avg_length,
            'tokens_per_second': throughput,
            'num_generations': num_generations
        }


class AsyncInferenceEngine(InferenceEngine):
    """
    Async version of the inference engine for high-concurrency serving
    """
    def __init__(self, model, tokenizer, config, max_concurrent_requests: int = 10):
        super().__init__(model, tokenizer, config)
        self.max_concurrent_requests = max_concurrent_requests
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.request_queue = asyncio.Queue()
        self.active_tasks = {}
    
    async def async_generate(self, inputs: str, **kwargs) -> str:
        """
        Async version of generate
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.generate(inputs, **kwargs)
        )
    
    async def process_batch_async(self, inputs: List[str], **kwargs) -> List[str]:
        """
        Process batch asynchronously
        """
        tasks = [self.async_generate(inp, **kwargs) for inp in inputs]
        results = await asyncio.gather(*tasks)
        return results


class ModelServer:
    """
    Production-ready model server with load balancing and health checks
    """
    def __init__(self, model, tokenizer, config, port: int = 8000, host: str = "0.0.0.0"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.port = port
        self.host = host
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(model, tokenizer, config)
        
        # Server stats
        self.stats = {
            'requests_processed': 0,
            'total_processing_time': 0,
            'active_requests': 0
        }
        
        # Health status
        self.health_status = {'status': 'healthy', 'timestamp': time.time()}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the server"""
        return self.health_status
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        avg_processing_time = (
            self.stats['total_processing_time'] / self.stats['requests_processed'] 
            if self.stats['requests_processed'] > 0 else 0
        )
        
        return {
            **self.stats,
            'avg_processing_time': avg_processing_time
        }
    
    def update_stats(self, processing_time: float):
        """Update server statistics"""
        self.stats['requests_processed'] += 1
        self.stats['total_processing_time'] += processing_time
    
    def perform_health_check(self) -> bool:
        """Perform a health check"""
        try:
            # Simple generation test
            test_input = "Hello"
            start_time = time.time()
            _ = self.inference_engine.generate(test_input, max_new_tokens=5)
            end_time = time.time()
            
            # Update health status
            self.health_status = {
                'status': 'healthy',
                'response_time': end_time - start_time,
                'timestamp': time.time()
            }
            return True
        except Exception as e:
            self.health_status = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
            return False


def create_api_server(model_path: str, tokenizer_path: str, config_path: str):
    """
    Create a FastAPI server for the model
    """
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException, Request
        from pydantic import BaseModel
        from typing import Optional, List
        
        # Load model and tokenizer
        from model.transformer import Transformer, Config
        config = Config.load(config_path)
        model = Transformer(config)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create server
        server = ModelServer(model, tokenizer, config)
        app = FastAPI(title="Transformer Model API", version="1.0.0")
        
        # Define request models
        class GenerateRequest(BaseModel):
            inputs: Union[str, List[str]]
            temperature: float = 0.8
            top_k: int = 50
            top_p: float = 0.95
            repetition_penalty: float = 1.2
            max_new_tokens: int = 100
            do_sample: bool = True
        
        class GenerateResponse(BaseModel):
            generated_text: Union[str, List[str]]
            processing_time: float
        
        # API endpoints
        @app.get("/")
        def root():
            return {"message": "Transformer Model API", "status": "ready"}
        
        @app.get("/health")
        def health():
            return server.get_health_status()
        
        @app.get("/stats")
        def stats():
            return server.get_server_stats()
        
        @app.post("/generate", response_model=GenerateResponse)
        def generate(request: GenerateRequest):
            start_time = time.time()
            try:
                result = server.inference_engine.generate(
                    inputs=request.inputs,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=request.do_sample
                )
                
                processing_time = time.time() - start_time
                server.update_stats(processing_time)
                
                return GenerateResponse(
                    generated_text=result,
                    processing_time=processing_time
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
        
        # Run server
        uvicorn.run(app, host=server.host, port=server.port)
        
    except ImportError:
        print("FastAPI and uvicorn are required for the API server: pip install fastapi uvicorn")


def load_model_for_inference(model_path: str, config_path: str, tokenizer_path: str):
    """
    Load model specifically for inference with optimizations
    """
    from model.transformer import Transformer, Config
    
    # Load configuration
    config = Config.load(config_path)
    
    # Load model
    model = Transformer(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    
    # Set to evaluation mode
    model.eval()
    
    # Apply optimizations for inference
    model = torch.jit.optimize_for_inference(torch.jit.script(model))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer, config


async def serve_with_gradio(model, tokenizer, config, share: bool = False):
    """
    Serve the model using Gradio for quick demos
    """
    try:
        import gradio as gr
        
        engine = InferenceEngine(model, tokenizer, config)
        
        def generate_response(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
            result = engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            return result
        
        gr.Interface(
            fn=generate_response,
            inputs=[
                gr.Textbox(label="Input Prompt", lines=3),
                gr.Slider(minimum=1, maximum=500, value=100, label="Max New Tokens"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
            ],
            outputs=gr.Textbox(label="Generated Text"),
            title="Transformer Model Demo",
            description="Enter a prompt to generate text with the transformer model."
        ).launch(share=share, server_port=7860)
        
    except ImportError:
        print("Gradio is required for the demo interface: pip install gradio")