import sys
import cv2
from pathlib import Path

# Add project root to PYTHONPATH
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

try:
    from server.utils.random_sampler import get_random_videos
    from server.preprocessing.preprocess_video import extract_frames
except ImportError as e:
    print(f"Import Error: {e}")
    get_random_videos = None
    extract_frames = None

#=====================================
# Test cv2 installation
#=====================================
def test_cv2_installation():
    """Test if OpenCV is properly installed."""
    try:
        version = cv2.__version__
        print(f"✓ OpenCV version: {version}")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

#=====================================
# Test random video sampling
#=====================================
def test_random_video_sampling():
    """Test random video sampling function."""
    if get_random_videos is None:
        print("✗ get_random_videos not imported")
        return False
    
    try:
        videos = get_random_videos(5)
        print(f"✓ Retrieved {len(videos)} random videos")
        if videos:
            print(f"  First 2 samples: {videos[:2]}")
        else:
            print("  No videos found")
        return True
    except Exception as e:
        print(f"✗ Random sampling test failed: {e}")
        return False

#=====================================
# Test frame extraction (optional - requires video file)
#=====================================
def test_frame_extraction(video_path=None):
    """Test video frame extraction function."""
    if extract_frames is None:
        print("✗ extract_frames not imported")
        return False
    
    if not video_path:
        print("✗ No video path provided for frame extraction test")
        print("  Usage: python test_script.py frames <video_path>")
        return False
    
    try:
        frames = extract_frames(video_path, frame_count=10)
        print(f"✓ Extracted {len(frames)} frames from {video_path}")
        return True
    except FileNotFoundError:
        print(f"✗ Video file not found: {video_path}")
        return False
    except Exception as e:
        print(f"✗ Frame extraction test failed: {e}")
        return False

#=====================================
# Helper function to show usage
#=====================================
def show_usage():
    """Display usage instructions."""
    print("\n" + "="*50)
    print("TEST SCRIPT USAGE")
    print("="*50)
    print("python test_script.py [test_name] [optional_args]")
    print("\nAvailable tests:")
    print("  cv2                    - Test OpenCV installation")
    print("  sampling               - Test random video sampling")
    print("  frames <video_path>    - Test frame extraction")
    print("  all                    - Run all tests (except frames)")
    print("  help                   - Show this usage message")
    print("\nExamples:")
    print("  python test_script.py cv2")
    print("  python test_script.py sampling")
    print("  python test_script.py frames test_video.mp4")
    print("  python test_script.py all")
    print("  python test_script.py          # Runs cv2 and sampling")
    print("="*50 + "\n")

#=====================================
# Main execution with validation
#=====================================
if __name__ == "__main__":
    import sys
    
    # Get command line arguments
    args = sys.argv[1:]  # Exclude script name
    
    # Track test results for summary
    test_results = {}
    
    try:
        if not args:
            # No arguments: run default tests
            print("\nRunning default tests (cv2 and sampling)...\n")
            test_results['cv2'] = test_cv2_installation()
            print()
            test_results['sampling'] = test_random_video_sampling()
            
        elif args[0] == "help":
            show_usage()
            sys.exit(0)
            
        elif args[0] == "cv2":
            print("\nRunning OpenCV installation test...\n")
            test_results['cv2'] = test_cv2_installation()
            
        elif args[0] == "sampling":
            print("\nRunning random video sampling test...\n")
            test_results['sampling'] = test_random_video_sampling()
            
        elif args[0] == "frames":
            if len(args) >= 2:
                video_path = args[1]
                print(f"\nRunning frame extraction test on: {video_path}\n")
                test_results['frames'] = test_frame_extraction(video_path)
            else:
                print("\n✗ Error: Missing video path for frames test")
                print("  Usage: python test_script.py frames <video_path>\n")
                show_usage()
                sys.exit(1)
                
        elif args[0] == "all":
            print("\nRunning all tests...\n")
            test_results['cv2'] = test_cv2_installation()
            print()
            test_results['sampling'] = test_random_video_sampling()
            print()
            print("Note: Skipping frames test (requires video path)")
            print("      Use 'python test_script.py frames <video_path>' for frame extraction")
            
        else:
            print(f"\n✗ Error: Unknown test '{args[0]}'\n")
            show_usage()
            sys.exit(1)
        
        # Print summary if any tests were run
        if test_results:
            print("\n" + "="*50)
            print("TEST SUMMARY")
            print("="*50)
            passed = sum(1 for result in test_results.values() if result)
            total = len(test_results)
            
            for test_name, result in test_results.items():
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"{test_name:15} {status}")
            
            print("-"*50)
            print(f"TOTAL: {passed}/{total} tests passed")
            print("="*50)
            
            # Exit with appropriate code (0 for success, 1 for failure)
            sys.exit(0 if passed == total else 1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)