#!/usr/bin/env python3
"""
Simple OpenAI Chart Similarity Test Script
Run this to test chart analysis without FastAPI/frontend
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
import json
import base64
from openai import OpenAI
import logging
from PIL import Image
import io
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI client setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)

def create_side_by_side_image(left_image_path, right_image_path):
    """Create a side-by-side comparison image with labels"""
    try:
        # Open both images
        left_img = Image.open(left_image_path)
        right_img = Image.open(right_image_path)
        
        # Convert to RGB if needed
        if left_img.mode != 'RGB':
            left_img = left_img.convert('RGB')
        if right_img.mode != 'RGB':
            right_img = right_img.convert('RGB')
        
        # Resize to consistent size for better analysis
        target_size = (500, 400)  # Larger size for better clarity
        left_img = left_img.resize(target_size, Image.Resampling.LANCZOS)
        right_img = right_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create side-by-side image with labels
        padding = 30
        label_height = 50
        text_size = 24
        
        combined_width = target_size[0] * 2 + padding * 3
        combined_height = target_size[1] + label_height + padding * 2
        
        # Create white background
        combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
        
        # Try to add text labels (optional, will work without if no font)
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_img)
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", text_size)
            except:
                font = ImageFont.load_default()
            
            # Add labels
            draw.text((padding + target_size[0]//2 - 50, 10), "LEFT CHART", 
                     fill='black', font=font)
            draw.text((padding * 2 + target_size[0] + target_size[0]//2 - 60, 10), "RIGHT CHART", 
                     fill='black', font=font)
        except ImportError:
            # Skip labels if PIL doesn't have text support
            pass
        
        # Paste images
        left_x = padding
        left_y = label_height
        combined_img.paste(left_img, (left_x, left_y))
        
        right_x = target_size[0] + padding * 2
        right_y = label_height
        combined_img.paste(right_img, (right_x, right_y))
        
        # Convert to base64 with high quality
        img_bytes = io.BytesIO()
        combined_img.save(img_bytes, format='JPEG', quality=95, optimize=True)
        img_bytes.seek(0)
        
        base64_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # Verify the base64 string is valid
        if len(base64_str) < 1000:  # Too small, probably failed
            logger.error("Generated base64 string too small, image processing may have failed")
            return None
            
        return base64_str
        
    except Exception as e:
        logger.error(f"Error creating side-by-side image: {e}")
        return None

def analyze_frame_similarity(left_path, right_path, frame_num, frame_time):
    """Analyze similarity between two chart images with improved prompting"""
    try:
        print(f"\n🔍 Analyzing Frame {frame_num} at {frame_time:.1f}s...")
        
        # Verify both image files exist and are readable
        if not left_path.exists() or not right_path.exists():
            print("❌ Image files not found")
            return None
            
        # Test that images can be opened
        try:
            with Image.open(left_path) as test_img:
                if test_img.size[0] < 10 or test_img.size[1] < 10:
                    print("❌ Left image too small")
                    return None
            with Image.open(right_path) as test_img:
                if test_img.size[0] < 10 or test_img.size[1] < 10:
                    print("❌ Right image too small")
                    return None
        except Exception as e:
            print(f"❌ Cannot open image files: {e}")
            return None
        
        # Create side-by-side image
        combined_b64 = create_side_by_side_image(left_path, right_path)
        if not combined_b64:
            print("❌ Failed to create combined image")
            return None
        
        print(f"✅ Created combined image ({len(combined_b64)} chars)")
        
        # Enhanced prompt that forces the AI to analyze the image
        prompt = """
        IMPORTANT: You can see two financial charts displayed side by side in this image.
        
        LEFT CHART: The chart on the left side of the image
        RIGHT CHART: The chart on the right side of the image
        
        I need you to VISUALLY COMPARE these two charts that you can see in the image.
        
        Look at the actual price movements, trends, and patterns you can observe.
        
        Rate their similarity from 0.0 to 1.0 based on:
        1. Overall trend direction (up/down/sideways)
        2. Pattern shapes and formations you can see
        3. Volatility levels visible in the charts
        4. Visual similarity of the price movements
        
        You MUST respond in this EXACT format:
        SIMILARITY_SCORE: [0.0-1.0]
        ANALYSIS: [describe what you actually see in both charts and how they compare]
        
        Do NOT say you cannot see the images - analyze what you observe in the charts shown.
        """
        
        print("📡 Calling OpenAI API...")
        start_time = time.time()
        
        # Call OpenAI with explicit instructions
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a financial chart analyst. You can see and analyze images of financial charts. When given chart images, you must analyze what you observe and provide detailed comparisons."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{combined_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=400,  # More tokens for detailed analysis
            temperature=0.1
        )
        
        duration = time.time() - start_time
        response_text = response.choices[0].message.content
        
        # Check if response contains "unable" or similar phrases
        if any(phrase in response_text.lower() for phrase in [
            "unable to", "can't see", "cannot see", "can't view", "cannot view",
            "unable to view", "unable to see", "no image", "cannot analyze"
        ]):
            print(f"⚠️  AI claims unable to see image, retrying with different approach...")
            
            # Retry with simpler prompt
            simple_prompt = """
            Analyze the two financial charts shown in this image. 
            Give them a similarity score from 0.0 to 1.0.
            
            Format:
            SIMILARITY_SCORE: [score]
            ANALYSIS: [comparison]
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": simple_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{combined_b64}",
                                "detail": "auto"
                            }
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content
            duration = time.time() - start_time
        
        # Extract similarity score
        score = extract_similarity_score(response_text)
        analysis = extract_analysis_text(response_text)
        
        print(f"✅ API Response ({duration:.1f}s):")
        print(f"   Similarity: {score:.3f}")
        print(f"   Analysis: {analysis[:150]}...")
        
        # Check for problematic responses
        if "unable" in analysis.lower() or score == 0.5:
            print(f"⚠️  Potentially problematic response detected")
            print(f"   Full response: {response_text[:200]}...")
        
        return {
            'frame_number': frame_num,
            'time': frame_time,
            'similarity': score,
            'analysis': analysis,
            'api_duration': duration,
            'raw_response': response_text,
            'image_size': len(combined_b64)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing frame {frame_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_similarity_score(response_text):
    """Extract similarity score from response"""
    import re
    
    patterns = [
        r'SIMILARITY_SCORE:\s*([0-1](?:\.[0-9]+)?)',
        r'similarity.*?score.*?([0-1](?:\.[0-9]+)?)',
        r'score.*?([0-1](?:\.[0-9]+)?)',
        r'\b([0-1]\.[0-9]+)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
    
    # Fallback based on keywords
    lower = response_text.lower()
    if 'very similar' in lower or 'identical' in lower:
        return 0.9
    elif 'similar' in lower:
        return 0.7
    elif 'somewhat' in lower:
        return 0.5
    elif 'different' in lower:
        return 0.2
    else:
        return 0.5

def extract_analysis_text(response_text):
    """Extract analysis text from response"""
    import re
    
    analysis_match = re.search(r'ANALYSIS:\s*(.*)', response_text, re.DOTALL | re.IGNORECASE)
    if analysis_match:
        return analysis_match.group(1).strip()
    
    # Clean up response
    cleaned = re.sub(r'SIMILARITY_SCORE:.*?\n', '', response_text, flags=re.IGNORECASE)
    return cleaned.strip()

def process_video(video_path, output_dir, fps=0.5):
    """Process video and analyze chart similarity"""
    print(f"🎬 Processing video: {video_path}")
    print(f"📊 Target FPS: {fps}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps_video
    
    print(f"📹 Video info: {total_frames} frames, {fps_video:.1f} fps, {duration:.1f}s")
    
    # Calculate frame interval
    frame_interval = int(fps_video / fps)
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"🔄 Processing every {frame_interval} frames")
    
    # Results storage
    results = []
    frame_num = 0
    processed_count = 0
    
    print("\n" + "="*60)
    print("🚀 STARTING ANALYSIS")
    print("="*60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame at intervals
        if frame_num % frame_interval == 0:
            frame_time = frame_num / fps_video
            
            # Save original frame
            frame_filename = f"frame_{frame_num:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Split frame in half
            height, width = frame.shape[:2]
            mid_point = width // 2
            left_frame = frame[:, :mid_point]
            right_frame = frame[:, mid_point:]
            
            # Save split frames
            left_filename = f"left_{frame_num:06d}.jpg"
            right_filename = f"right_{frame_num:06d}.jpg"
            left_path = output_dir / left_filename
            right_path = output_dir / right_filename
            cv2.imwrite(str(left_path), left_frame)
            cv2.imwrite(str(right_path), right_frame)
            
            # Analyze similarity
            result = analyze_frame_similarity(left_path, right_path, frame_num, frame_time)
            
            if result:
                results.append(result)
                processed_count += 1
                
                print(f"📈 Frame {frame_num}: {result['similarity']:.3f} similarity")
            else:
                print(f"⚠️  Frame {frame_num}: Analysis failed")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        frame_num += 1
    
    cap.release()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    # Sort results by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total frames processed: {processed_count}")
    print(f"   Video duration: {duration:.1f}s")
    print(f"   Average similarity: {np.mean([r['similarity'] for r in results]):.3f}")
    
    print(f"\n🏆 TOP 5 MOST SIMILAR MOMENTS:")
    for i, result in enumerate(results[:5]):
        print(f"   #{i+1}: Frame {result['frame_number']} at {result['time']:.1f}s - {result['similarity']:.3f}")
    
    # Save results to JSON
    output_file = output_dir / "results.json"
    final_results = {
        'video_path': str(video_path),
        'total_frames_processed': processed_count,
        'video_duration': duration,
        'analysis_fps': fps,
        'top_frames': results[:10],
        'all_results': results,
        'summary': {
            'max_similarity': max([r['similarity'] for r in results]) if results else 0,
            'min_similarity': min([r['similarity'] for r in results]) if results else 0,
            'avg_similarity': np.mean([r['similarity'] for r in results]) if results else 0
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📁 Frame images saved to: {output_dir}")
    
    return final_results

def main():
    """Main function"""
    print("🤖 OpenAI Chart Similarity Test")
    print("="*40)
    
    # Get video path from command line or input
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("📹 Enter video path: ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Set output directory
    video_name = Path(video_path).stem
    output_dir = f"analysis_{video_name}_{int(time.time())}"
    
    # Set analysis FPS
    fps = float(input("⚡ Enter analysis FPS (0.1-2.0, recommended 0.5): ") or "0.5")
    
    print(f"\n🎯 Configuration:")
    print(f"   Video: {video_path}")
    print(f"   Output: {output_dir}")
    print(f"   FPS: {fps}")
    print(f"   OpenAI Model: gpt-4o")
    
    confirm = input("\n🚀 Start analysis? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Analysis cancelled")
        return
    
    # Run analysis
    try:
        results = process_video(video_path, output_dir, fps)
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📄 Check {output_dir}/results.json for full results")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()