#!/usr/bin/env python3
"""
Master script to fetch all social media data for financial analysis
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    try:
        logger.info(f"Starting: {description}")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"✅ Completed: {description}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ Failed: {description}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {description} took too long")
        return False
    except Exception as e:
        logger.error(f"💥 Exception in {description}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'praw',           # Reddit API
        'tweepy',         # Twitter API
        'textblob',       # Sentiment analysis
        'vaderSentiment', # VADER sentiment
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_api_setup():
    """Check if API credentials are configured"""
    api_status = {
        'reddit': False,
        'twitter': False
    }
    
    # Check Reddit API
    reddit_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    if all(os.getenv(var) for var in reddit_vars):
        api_status['reddit'] = True
    
    # Check Twitter API
    twitter_vars = ['TWITTER_BEARER_TOKEN']
    twitter_v1_vars = ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET']
    
    if os.getenv('TWITTER_BEARER_TOKEN') or all(os.getenv(var) for var in twitter_v1_vars):
        api_status['twitter'] = True
    
    return api_status

def main():
    """Main function to fetch all social media data"""
    print("🚀 Financial Social Media Data Fetcher")
    print("="*50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    print("✅ All dependencies available")
    
    # Check API setup
    print("\n🔑 Checking API credentials...")
    api_status = check_api_setup()
    
    if not any(api_status.values()):
        print("❌ No API credentials configured!")
        print("\nPlease set up API credentials:")
        print("\n📱 For Reddit API:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new app (type: script)")
        print("3. Set environment variables:")
        print("   export REDDIT_CLIENT_ID='your_client_id'")
        print("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        
        print("\n🐦 For Twitter API:")
        print("1. Go to https://developer.twitter.com/")
        print("2. Create a new app")
        print("3. Set environment variable:")
        print("   export TWITTER_BEARER_TOKEN='your_bearer_token'")
        
        return
    
    print(f"Reddit API: {'✅' if api_status['reddit'] else '❌'}")
    print(f"Twitter API: {'✅' if api_status['twitter'] else '❌'}")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Track success of each step
    results = {}
    
    print(f"\n🎯 Starting social media data collection...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch Reddit data
    if api_status['reddit']:
        print(f"\n" + "="*50)
        results['reddit'] = run_script('reddit_fetcher.py', 'Fetching Reddit financial discussions')
    else:
        print(f"\n⏭️  Skipping Reddit (no API credentials)")
        results['reddit'] = False
    
    # Fetch Twitter data
    if api_status['twitter']:
        print(f"\n" + "="*50)
        results['twitter'] = run_script('twitter_fetcher.py', 'Fetching Twitter financial posts')
    else:
        print(f"\n⏭️  Skipping Twitter (no API credentials)")
        results['twitter'] = False
    
    # Analyze sentiment
    if any(results.values()):
        print(f"\n" + "="*50)
        results['sentiment'] = run_script('social_media_analyzer.py', 'Analyzing social media sentiment')
    else:
        print(f"\n⏭️  Skipping sentiment analysis (no social data)")
        results['sentiment'] = False
    
    # Integrate with training data
    if results.get('sentiment', False):
        print(f"\n" + "="*50)
        results['integration'] = run_script('social_data_integration.py', 'Integrating social data with training dataset')
    else:
        print(f"\n⏭️  Skipping integration (no sentiment data)")
        results['integration'] = False
    
    # Final summary
    print(f"\n" + "="*60)
    print("📊 SOCIAL MEDIA DATA COLLECTION SUMMARY")
    print("="*60)
    
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📈 Results:")
    
    step_names = {
        'reddit': 'Reddit data fetching',
        'twitter': 'Twitter data fetching', 
        'sentiment': 'Sentiment analysis',
        'integration': 'Training data integration'
    }
    
    successful_steps = 0
    for step, success in results.items():
        status = "✅ Success" if success else "❌ Failed/Skipped"
        print(f"  {step_names.get(step, step)}: {status}")
        if success:
            successful_steps += 1
    
    print(f"\n🎯 Overall: {successful_steps}/{len(results)} steps completed successfully")
    
    # Provide next steps
    if results.get('integration', False):
        print(f"\n🚀 Next Steps:")
        print(f"1. Train models with social data:")
        print(f"   python train_advanced.py --experiment social_enhanced")
        print(f"2. Check enhanced training data:")
        print(f"   ls -la data/training_data_with_social.csv")
        print(f"3. View social sentiment analysis:")
        print(f"   ls -la data/*enhanced.csv")
    
    elif any(results.values()):
        print(f"\n📝 Recommendations:")
        print(f"1. Check collected data in data/ folder")
        print(f"2. Run sentiment analysis manually if needed")
        print(f"3. Ensure training_data.csv exists for integration")
    
    else:
        print(f"\n❗ No data collected. Please:")
        print(f"1. Set up API credentials")
        print(f"2. Check internet connection")
        print(f"3. Review error messages above")

if __name__ == "__main__":
    main()