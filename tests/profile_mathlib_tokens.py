import os
import numpy as np
import statistics
from typing import List, Dict, Tuple
import tiktoken
from tqdm import tqdm
import json
from pathlib import Path

class MathlibTokenProfiler:
    def __init__(self, mathlib_path: str = "mathlib4/Mathlib"):
        """Initialize the profiler with the mathlib directory path"""
        self.mathlib_path = mathlib_path
        
        # Initialize tokenizer (using the same tokenizer as OpenAI's text-embedding models)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print(f"Profiling files in: {mathlib_path}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.tokenizer.encode(text))
    
    def get_all_lean_files(self) -> List[str]:
        """Get all .lean files in the mathlib directory"""
        lean_files = []
        for root, _, files in os.walk(self.mathlib_path):
            for file in files:
                if file.endswith('.lean'):
                    lean_files.append(os.path.join(root, file))
        return lean_files
    
    def profile_files(self) -> Dict:
        """Profile all .lean files and calculate token statistics"""
        lean_files = self.get_all_lean_files()
        print(f"Found {len(lean_files)} .lean files")
        
        file_stats = []
        token_counts = []
        failed_files = []
        
        for file_path in tqdm(lean_files, desc="Processing .lean files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                token_count = self.count_tokens(content)
                relative_path = os.path.relpath(file_path, self.mathlib_path)
                
                file_info = {
                    'file_path': relative_path,
                    'absolute_path': file_path,
                    'token_count': token_count,
                    'character_count': len(content),
                    'line_count': len(content.splitlines())
                }
                
                file_stats.append(file_info)
                token_counts.append(token_count)
                
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"Error processing {file_path}: {str(e)}")
        
        # Calculate overall statistics
        if token_counts:
            overall_stats = {
                'total_files': len(token_counts),
                'total_tokens': sum(token_counts),
                'average_tokens': statistics.mean(token_counts),
                'median_tokens': statistics.median(token_counts),
                'max_tokens': max(token_counts),
                'min_tokens': min(token_counts),
                'std_deviation': statistics.stdev(token_counts) if len(token_counts) > 1 else 0
            }
            
            # Calculate percentiles
            token_distribution = {
                'percentile_25': np.percentile(token_counts, 25),
                'percentile_75': np.percentile(token_counts, 75),
                'percentile_90': np.percentile(token_counts, 90),
                'percentile_95': np.percentile(token_counts, 95),
                'percentile_99': np.percentile(token_counts, 99)
            }
        else:
            overall_stats = {}
            token_distribution = {}
        
        return {
            'overall_stats': overall_stats,
            'file_stats': file_stats,
            'token_distribution': token_distribution,
            'failed_files': failed_files
        }
    
    def get_largest_files(self, file_stats: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get the largest files by token count"""
        sorted_files = sorted(file_stats, key=lambda x: x['token_count'], reverse=True)
        return sorted_files[:top_n]
    
    def get_smallest_files(self, file_stats: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get the smallest files by token count"""
        sorted_files = sorted(file_stats, key=lambda x: x['token_count'])
        return sorted_files[:top_n]
    
    def analyze_by_directory(self, file_stats: List[Dict]) -> Dict:
        """Analyze token statistics by directory"""
        dir_stats = {}
        
        for file_info in file_stats:
            file_path = file_info['file_path']
            directory = os.path.dirname(file_path)
            
            if directory not in dir_stats:
                dir_stats[directory] = {
                    'files': [],
                    'token_counts': []
                }
            
            dir_stats[directory]['files'].append(file_info)
            dir_stats[directory]['token_counts'].append(file_info['token_count'])
        
        # Calculate statistics for each directory
        for directory, data in dir_stats.items():
            token_counts = data['token_counts']
            dir_stats[directory]['stats'] = {
                'file_count': len(token_counts),
                'total_tokens': sum(token_counts),
                'average_tokens': statistics.mean(token_counts),
                'median_tokens': statistics.median(token_counts),
                'max_tokens': max(token_counts),
                'min_tokens': min(token_counts)
            }
        
        return dir_stats
    
    def save_profile_report(self, profile_data: Dict, output_file: str = "mathlib_file_token_profile.json"):
        """Save the profiling results to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        print(f"Profile report saved to: {output_file}")

def print_summary(profile_data: Dict):
    """Print a summary of the profiling results"""
    overall = profile_data['overall_stats']
    
    if not overall:
        print("No files were successfully processed.")
        return
    
    print("\n" + "="*60)
    print("MATHLIB FILE TOKEN PROFILING SUMMARY")
    print("="*60)
    
    print(f"Total Files: {overall['total_files']:,}")
    print(f"Total Tokens: {overall['total_tokens']:,}")
    print(f"Average Tokens per File: {overall['average_tokens']:.2f}")
    print(f"Median Tokens per File: {overall['median_tokens']:.2f}")
    print(f"Maximum Tokens in File: {overall['max_tokens']:,}")
    print(f"Minimum Tokens in File: {overall['min_tokens']:,}")
    print(f"Standard Deviation: {overall['std_deviation']:.2f}")
    
    print("\nToken Distribution Percentiles:")
    dist = profile_data['token_distribution']
    print(f"25th Percentile: {dist['percentile_25']:.2f}")
    print(f"75th Percentile: {dist['percentile_75']:.2f}")
    print(f"90th Percentile: {dist['percentile_90']:.2f}")
    print(f"95th Percentile: {dist['percentile_95']:.2f}")
    print(f"99th Percentile: {dist['percentile_99']:.2f}")

def main():
    # Initialize profiler
    profiler = MathlibTokenProfiler()
    
    # Profile all files
    profile_data = profiler.profile_files()
    
    # Print summary
    print_summary(profile_data)
    
    if profile_data['overall_stats']:
        # Get largest files
        print("\n" + "="*60)
        print("TOP 10 LARGEST FILES BY TOKEN COUNT")
        print("="*60)
        
        largest_files = profiler.get_largest_files(profile_data['file_stats'], top_n=10)
        for i, file_info in enumerate(largest_files, 1):
            print(f"{i:2d}. {file_info['file_path']}")
            print(f"    Tokens: {file_info['token_count']:,}")
            print(f"    Characters: {file_info['character_count']:,}")
            print(f"    Lines: {file_info['line_count']:,}")
            print()
        
        # Get smallest files
        print("="*60)
        print("TOP 10 SMALLEST FILES BY TOKEN COUNT")
        print("="*60)
        
        smallest_files = profiler.get_smallest_files(profile_data['file_stats'], top_n=10)
        for i, file_info in enumerate(smallest_files, 1):
            print(f"{i:2d}. {file_info['file_path']}")
            print(f"    Tokens: {file_info['token_count']:,}")
            print(f"    Characters: {file_info['character_count']:,}")
            print(f"    Lines: {file_info['line_count']:,}")
            print()
        
        # Analyze by directory
        print("="*60)
        print("TOP 10 DIRECTORIES BY TOTAL TOKEN COUNT")
        print("="*60)
        
        dir_stats = profiler.analyze_by_directory(profile_data['file_stats'])
        sorted_dirs = sorted(dir_stats.items(), key=lambda x: x[1]['stats']['total_tokens'], reverse=True)
        
        for i, (directory, data) in enumerate(sorted_dirs[:10], 1):
            stats = data['stats']
            print(f"{i:2d}. {directory if directory else '(root)'}")
            print(f"    Files: {stats['file_count']}")
            print(f"    Total Tokens: {stats['total_tokens']:,}")
            print(f"    Avg Tokens/File: {stats['average_tokens']:.2f}")
            print()
    
    # Report failed files
    if profile_data['failed_files']:
        print("="*60)
        print(f"FAILED TO PROCESS {len(profile_data['failed_files'])} FILES")
        print("="*60)
        for file_path, error in profile_data['failed_files']:
            print(f"  {file_path}: {error}")
    
    # Save detailed report
    profiler.save_profile_report(profile_data)
    
    print("\nProfiling complete!")

if __name__ == "__main__":
    main() 