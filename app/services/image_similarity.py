#!/usr/bin/env python3
"""
Image similarity service for comparing text with video thumbnails/previews
Uses CLIP models for multimodal text-image similarity
"""

import os
import requests
import warnings
import time
import gc
import threading
import logging
try:
    import psutil  # For memory monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import List, Dict, Optional
from PIL import Image
import io
import numpy as np
from loguru import logger
from app.config import config

# Suppress transformers warnings about slow processors
warnings.filterwarnings("ignore", message=".*slow.*processor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# Global model cache
_clip_model = None
_clip_processor = None
_model_load_fails = 0  # Track model loading failures
_max_load_retries = 3  # Maximum retries before giving up
_force_cpu_only = True  # Force CPU-only mode to avoid GPU issues

# Add embedding cache to avoid reprocessing same images/text
_image_embedding_cache = {}
_text_embedding_cache = {}
_cache_max_size = 100  # Limit cache size to prevent memory issues
_caching_enabled = True  # Can be disabled for testing or if memory is limited

# Rate limiting to prevent memory issues
_last_inference_time = 0
_inference_count = 0
INFERENCE_DELAY = 0.15  # Slightly increased delay for stability
MAX_BATCH_SIZE = 10    # Process in smaller batches

def check_image_similarity_dependencies() -> bool:
    """Check if image similarity dependencies are available"""
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        return True
    except ImportError:
        return False

# Set availability flag
IMAGE_SIMILARITY_AVAILABLE = check_image_similarity_dependencies()

def timeout_wrapper(timeout_seconds=30):
    """Thread-safe decorator to add timeout protection to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            class TimeoutException(Exception):
                pass
            
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            # Create and start thread
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            # Wait for completion or timeout
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # Thread is still running - timeout occurred
                logger.error(f"⏰ Function {func.__name__} timed out after {timeout_seconds} seconds")
                return 0.0  # Return default similarity score
            
            if exception[0]:
                # Exception occurred in the thread
                logger.error(f"❌ Exception in {func.__name__}: {exception[0]}")
                return 0.0
                
            return result[0] if result[0] is not None else 0.0
                
        return wrapper
    return decorator

def load_clip_model(model_name: str = "clip-vit-base-patch32"):
    """Load CLIP model for text-image similarity"""
    global _clip_model, _clip_processor, _model_load_fails
    
    # Check if we've had too many failures
    if _model_load_fails >= _max_load_retries:
        safe_log("error", f"❌ Maximum model loading retries ({_max_load_retries}) exceeded")
        raise Exception(f"Model loading failed {_model_load_fails} times, giving up")
    
    if _clip_model is not None:
        return _clip_model, _clip_processor
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        # Map model names to HuggingFace model IDs
        model_mapping = {
            "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
            "clip-vit-base-patch16": "openai/clip-vit-base-patch16", 
            "clip-vit-large-patch14": "openai/clip-vit-large-patch14"
        }
        
        hf_model_name = model_mapping.get(model_name, model_name)
        
        safe_log("info", f"🖼️  Loading CLIP model: {model_name}")
        
        # Set up cache directory for persistent storage
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load processor with slow tokenizer for compatibility
        try:
            _clip_processor = CLIPProcessor.from_pretrained(
                hf_model_name,
                cache_dir=cache_dir,
                use_fast=False  # Use slow processor to avoid CLIPImageProcessorFast attribute errors
            )
        except Exception as processor_error:
            safe_log("warning", f"⚠️  Could not load processor with cache ({processor_error}), trying without cache")
            _clip_processor = CLIPProcessor.from_pretrained(
                hf_model_name,
                use_fast=False
            )
        
        # Try to load with safetensors first, fallback to regular loading
        try:
            _clip_model = CLIPModel.from_pretrained(
                hf_model_name,
                cache_dir=cache_dir,
                use_safetensors=True
            )
            safe_log("info", "✅ Loaded model using safetensors format")
        except Exception as safetensor_error:
            safe_log("warning", f"⚠️  Could not load with safetensors ({safetensor_error}), falling back to regular loading")
            _clip_model = CLIPModel.from_pretrained(
                hf_model_name,
                cache_dir=cache_dir
            )
            safe_log("info", "✅ Loaded model using regular format")
        
        # Keep model on CPU to avoid memory issues
        _clip_model = _clip_model.to("cpu")
        
        # Force CPU-only mode if enabled
        if _force_cpu_only:
            safe_log("info", "🖥️  Forcing CPU-only mode for CLIP model")
            _clip_model = _clip_model.to("cpu")
            # Ensure no CUDA operations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.enabled = False
        
        safe_log("success", f"✅ CLIP model loaded successfully: {model_name} (device: {'CPU-only' if _force_cpu_only else 'auto'})")
        
        # Reset failure count on successful load
        _model_load_fails = 0
        
        return _clip_model, _clip_processor
        
    except Exception as e:
        _model_load_fails += 1
        safe_log("error", f"Failed to load CLIP model {model_name} (attempt {_model_load_fails}/{_max_load_retries}): {e}")
        
        # If we have failures, try resetting the model
        if _model_load_fails < _max_load_retries:
            reset_clip_model()
        
        raise

@timeout_wrapper(timeout_seconds=30)  # Re-enable timeout protection
def calculate_text_image_similarity(text: str, image_url: str, model_name: str = "clip-vit-base-patch32") -> float:
    """Calculate similarity between text and image using CLIP"""
    global _last_inference_time, _inference_count, _image_embedding_cache, _text_embedding_cache
    
    if not IMAGE_SIMILARITY_AVAILABLE:
        safe_log("warning", "Image similarity not available - missing dependencies")
        return 0.0
    
    try:
        # Minimal logging - only log device info once per session
        if not hasattr(calculate_text_image_similarity, '_device_logged'):
            import torch
            device_info = "CPU-only" if _force_cpu_only else ("CUDA" if torch.cuda.is_available() else "CPU")
            safe_log("info", f"🖥️  Image similarity mode: {device_info}")
            calculate_text_image_similarity._device_logged = True
        
        import torch
        
        # Check cache first
        text_cache_key = f"{model_name}:{text}"
        image_cache_key = f"{model_name}:{image_url}"
        
        # If both embeddings are cached, calculate similarity directly
        if _caching_enabled and text_cache_key in _text_embedding_cache and image_cache_key in _image_embedding_cache:
            try:
                text_embeds = _text_embedding_cache[text_cache_key]
                image_embeds = _image_embedding_cache[image_cache_key]
                
                with torch.no_grad():
                    similarity = torch.cosine_similarity(text_embeds, image_embeds, dim=-1).item()
                    similarity = (similarity + 1) / 2  # Convert from (-1, 1) to (0, 1)
                    
                return float(similarity)
            except Exception as cache_error:
                safe_log("error", f"❌ Error calculating from cache: {cache_error}")
                # Continue with fresh calculation
        
        # Rate limiting to prevent overwhelming the system
        current_time = time.time()
        if current_time - _last_inference_time < INFERENCE_DELAY:
            time.sleep(INFERENCE_DELAY)
        _last_inference_time = time.time()
        _inference_count += 1
        
        # Periodic cleanup every 200 inferences - minimal logging
        if _inference_count % 200 == 0:
            try:
                clear_cache_if_needed()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as cleanup_error:
                safe_log("error", f"❌ Error during cleanup: {cleanup_error}")
        
        # Load model - reduced logging
        try:
            model, processor = load_clip_model(model_name)
        except Exception as model_error:
            safe_log("error", f"❌ Error loading CLIP model: {model_error}")
            
            # Try to reset and reload model once
            try:
                safe_log("warning", "🔄 Attempting model reset and reload...")
                reset_clip_model()
                model, processor = load_clip_model(model_name)
                safe_log("success", "✅ Model reset and reload successful")
            except Exception as retry_error:
                safe_log("error", f"❌ Model reset and reload failed: {retry_error}")
                return 0.0
        
        # Download and process image with timeout and size limits (only if not cached)
        image_embeds = None
        image = None
        
        if _caching_enabled and image_cache_key in _image_embedding_cache:
            image_embeds = _image_embedding_cache[image_cache_key]
        
        # Always download image if we don't have cached embeddings or if we need it for processing
        if image_embeds is None or not _caching_enabled or text_cache_key not in _text_embedding_cache:
            try:
                # Use shorter timeout and add more robust error handling
                response = requests.get(
                    image_url, 
                    timeout=(5, 10),  # 5s connect, 10s read timeout
                    stream=True,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; ImageBot/1.0)'}
                )
                response.raise_for_status()
                
                # Check content size to prevent memory issues
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                    safe_log("warning", f"Image too large ({content_length} bytes), skipping: {image_url}")
                    return 0.0
                
                # Download with size limit
                image_data = b""
                downloaded = 0
                max_size = 10 * 1024 * 1024  # 10MB limit
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            safe_log("warning", f"Image download exceeded size limit, stopping at {downloaded} bytes")
                            break
                        image_data += chunk
                
                if len(image_data) == 0:
                    safe_log("warning", f"Empty image data received from: {image_url}")
                    return 0.0
                
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Resize large images to prevent memory issues
                max_size = 512
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
            except requests.exceptions.Timeout:
                safe_log("error", f"❌ Timeout downloading image from: {image_url}")
                return 0.0
            except requests.exceptions.ConnectionError:
                safe_log("error", f"❌ Connection error downloading image from: {image_url}")
                return 0.0
            except requests.exceptions.HTTPError as e:
                safe_log("error", f"❌ HTTP error downloading image from {image_url}: {e}")
                return 0.0
            except Exception as img_error:
                safe_log("error", f"❌ Failed to load image {image_url}: {img_error}")
                return 0.0
        
        # Process text (only if not cached)
        text_embeds = None
        if _caching_enabled and text_cache_key in _text_embedding_cache:
            text_embeds = _text_embedding_cache[text_cache_key]
        
        # Process inputs with error handling - minimal logging
        try:
            if text_embeds is None or image_embeds is None:
                # We need both text and image for the processor, so we'll process them together
                # but only cache what we don't have
                inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
                    
        except AttributeError as attr_error:
            if "_valid_processor_keys" in str(attr_error):
                safe_log("warning", f"⚠️  Processor compatibility issue ({attr_error}), reloading model...")
                # Clear global cache and reload
                global _clip_model, _clip_processor
                _clip_model = None
                _clip_processor = None
                model, processor = load_clip_model(model_name)
                inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
            else:
                safe_log("error", f"❌ Attribute error in processor: {attr_error}")
                raise
        except Exception as proc_error:
            safe_log("error", f"❌ Failed to process inputs: {proc_error}")
            return 0.0
        
        # Get embeddings with no_grad to save memory - minimal logging
        similarity = 0.0
        try:
            with torch.no_grad():
                if text_embeds is None or image_embeds is None:
                    
                    # Ensure inputs are on the correct device
                    if _force_cpu_only:
                        # Move all inputs to CPU explicitly
                        inputs = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    
                    if text_embeds is None:
                        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
                        # Cache the text embedding
                        if _caching_enabled:
                            _text_embedding_cache[text_cache_key] = text_embeds.clone()
                        
                    if image_embeds is None:
                        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                        # Cache the image embedding
                        if _caching_enabled:
                            _image_embedding_cache[image_cache_key] = image_embeds.clone()
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(text_embeds, image_embeds, dim=-1).item()
                
                # Convert from (-1, 1) to (0, 1) range
                similarity = (similarity + 1) / 2
                
        except Exception as inference_error:
            safe_log("error", f"❌ Failed during model inference: {inference_error}")
            import traceback
            safe_log("error", f"❌ Traceback: {traceback.format_exc()}")
            
            # Try to reset model if inference fails repeatedly
            global _model_load_fails
            if "CUDA" in str(inference_error) or "memory" in str(inference_error).lower():
                safe_log("warning", "🔄 Memory/CUDA error detected, resetting model...")
                try:
                    reset_clip_model()
                except Exception:
                    pass

            return 0.0
        finally:
            # Aggressive cleanup to prevent memory leaks - no debug logging
            try:
                if 'inputs' in locals():
                    del inputs
                if 'outputs' in locals():
                    del outputs
                if 'text_embeds' in locals() and text_cache_key not in _text_embedding_cache:
                    del text_embeds
                if 'image_embeds' in locals() and image_cache_key not in _image_embedding_cache:
                    del image_embeds
                if 'image' in locals():
                    del image
                if 'image_data' in locals():
                    del image_data
                    
                # Force garbage collection for this inference
                gc.collect()
                
                # Clear CUDA cache if available and not in CPU-only mode
                if torch.cuda.is_available() and not _force_cpu_only:
                    torch.cuda.empty_cache()
                    
            except Exception as cleanup_error:
                safe_log("error", f"❌ Cleanup warning: {cleanup_error}")
            
        return float(similarity)
            
    except Exception as e:
        safe_log("error", f"❌ Failed to calculate text-image similarity: {e}")
        import traceback
        safe_log("error", f"❌ Full traceback: {traceback.format_exc()}")
        # Emergency cleanup
        try:
            gc.collect()
            if 'torch' in locals():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass
        return 0.0

def calculate_video_image_similarity(text: str, video_metadata: Dict, model_name: str = "clip-vit-base-patch32") -> float:
    """Calculate similarity between text and video images (thumbnail + preview frames)"""
    if not IMAGE_SIMILARITY_AVAILABLE:
        return 0.0
        
    # Get image URLs from video metadata
    image_urls = []
    
    # Add thumbnail if available
    if 'thumbnail_url' in video_metadata and video_metadata['thumbnail_url']:
        image_urls.append(video_metadata['thumbnail_url'])
    
    # Add preview images if available
    if 'preview_images' in video_metadata and video_metadata['preview_images']:
        image_urls.extend(video_metadata['preview_images'])
    
    if not image_urls:
        return 0.0
    
    # Smart selection: pick the most representative images
    selected_images = select_representative_images(image_urls, max_images=1)
    
    # Calculate similarity for selected images and return the best match
    max_similarity = 0.0
    
    for url in selected_images:
        try:
            similarity = calculate_text_image_similarity(text, url, model_name)
            max_similarity = max(max_similarity, similarity)
            
        except Exception as e:
            # Only log errors for debugging failed image processing
            logger.warning(f"Failed to process image {url}: {e}")
            continue
    
    return max_similarity

def download_image(image_url: str) -> Optional[Image.Image]:
    """Download and load image from URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        
        response = requests.get(
            image_url, 
            headers=headers, 
            proxies=config.proxy,
            timeout=30
        )
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.warning(f"Failed to download image from {image_url}: {e}")
        return None

def select_representative_images(image_urls: List[str], max_images: int = 1) -> List[str]:
    """Select the most representative images from a list"""
    if not image_urls:
        return []
    
    if len(image_urls) <= max_images:
        return image_urls
    
    # Strategy: For single image, always use thumbnail (first image)
    # For multiple images: Take thumbnail + middle image(s) from preview frames
    selected = []
    
    # Always include thumbnail if available (usually first and most representative)
    selected.append(image_urls[0])
    
    # Only select additional images if max_images > 1
    if len(image_urls) > 1 and max_images > 1:
        remaining_slots = max_images - 1
        remaining_images = image_urls[1:]  # Skip thumbnail
        
        if remaining_images:
            # Select middle image(s)
            step = max(1, len(remaining_images) // (remaining_slots + 1))
            for i in range(remaining_slots):
                idx = min((i + 1) * step, len(remaining_images) - 1)
                if idx < len(remaining_images):
                    selected.append(remaining_images[idx])
    
    return selected[:max_images]

def clear_cache_if_needed():
    """Clear caches if they exceed maximum size"""
    global _image_embedding_cache, _text_embedding_cache
    
    if len(_image_embedding_cache) > _cache_max_size:
        # Remove oldest entries (simple FIFO)
        to_remove = len(_image_embedding_cache) - _cache_max_size + 10  # Remove a few extra
        for _ in range(to_remove):
            if _image_embedding_cache:
                _image_embedding_cache.popitem(last=False)
    
    if len(_text_embedding_cache) > _cache_max_size:
        # Remove oldest entries (simple FIFO)
        to_remove = len(_text_embedding_cache) - _cache_max_size + 10  # Remove a few extra
        for _ in range(to_remove):
            if _text_embedding_cache:
                _text_embedding_cache.popitem(last=False)

def clear_all_caches():
    """Clear all embedding caches"""
    global _image_embedding_cache, _text_embedding_cache
    
    logger.info(f"🧹 Clearing all caches (image: {len(_image_embedding_cache)}, text: {len(_text_embedding_cache)})")
    _image_embedding_cache.clear()
    _text_embedding_cache.clear()

def get_cache_stats():
    """Get cache statistics"""
    return {
        'text_cache_size': len(_text_embedding_cache),
        'image_cache_size': len(_image_embedding_cache),
        'cache_max_size': _cache_max_size,
        'caching_enabled': _caching_enabled,
        'inference_count': _inference_count,
        'model_load_fails': _model_load_fails
    }

def get_memory_usage():
    """Get current memory usage statistics"""
    if not PSUTIL_AVAILABLE:
        return {"error": "psutil not available"}
        
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "process_memory_percent": memory_percent,
            "system_memory_percent": system_memory.percent,
            "system_available_gb": system_memory.available / 1024 / 1024 / 1024
        }
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return {"error": str(e)}

def log_memory_usage(context: str = ""):
    """Log current memory usage for debugging - minimal logging"""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        memory = get_memory_usage()
        if memory.get('error'):
            # Only log errors, not regular monitoring
            logger.debug(f"🧠 Memory monitoring error {context}: {memory['error']}")
        # Remove regular memory usage debug logs to reduce log flood
    except Exception as e:
        logger.debug(f"🧠 Memory logging failed {context}: {e}")

def reset_clip_model():
    """Reset the global CLIP model if it gets into a bad state"""
    global _clip_model, _clip_processor, _model_load_fails
    safe_log("warning", "🔄 Resetting CLIP model due to errors")
    
    try:
        # Clear any existing model
        if _clip_model is not None:
            del _clip_model
        if _clip_processor is not None:
            del _clip_processor
            
        _clip_model = None
        _clip_processor = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        safe_log("info", "✅ CLIP model reset completed")
        
    except Exception as e:
        safe_log("error", f"❌ Error during model reset: {e}")
        _model_load_fails += 1

def is_model_healthy() -> bool:
    """Check if the model is in a healthy state"""
    global _model_load_fails, _clip_model, _clip_processor
    
    # Check if we've had too many failures
    if _model_load_fails >= _max_load_retries:
        return False
    
    # If model is not loaded yet, that's okay - we can try to load it
    # Only return False if we have a loaded model that's in a bad state
    return True

def force_model_reset():
    """Force a model reset - useful for debugging or when model gets stuck"""
    global _model_load_fails
    safe_log("info", "🔄 Forcing model reset...")
    _model_load_fails = 0  # Reset failure count
    reset_clip_model()
    safe_log("info", "✅ Forced model reset completed")

# Create a thread-safe logger for background operations
def safe_log(level: str, message: str):
    """Thread-safe logging that won't interfere with Streamlit"""
    import threading
    import logging
    
    # Check if we're in the main thread
    main_thread = threading.main_thread()
    current_thread = threading.current_thread()
    
    # If we're not in the main thread, use standard logging to avoid Streamlit issues
    if current_thread != main_thread:
        try:
            log_level = getattr(logging, level.upper(), logging.INFO)
            logging.log(log_level, message)
            return
        except Exception:
            # Ultimate fallback - just print
            print(f"[{level.upper()}] {message}")
            return
    
    # We're in the main thread, safe to use loguru
    try:
        if level.lower() == 'debug':
            logger.debug(message)
        elif level.lower() == 'info':
            logger.info(message)
        elif level.lower() == 'warning':
            logger.warning(message)
        elif level.lower() == 'error':
            logger.error(message)
        elif level.lower() == 'success':
            logger.success(message)
        else:
            logger.info(message)
    except Exception as safe_log_error:
        # Ultimate fallback if loguru fails
        try:
            log_level = getattr(logging, level.upper(), logging.INFO)
            logging.log(log_level, message)
        except Exception:
            print(f"[{level.upper()}] {message}")