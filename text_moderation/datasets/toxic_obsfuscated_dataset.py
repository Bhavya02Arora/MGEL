"""
Generate CSV with toxic obfuscated words for training
Saves to: datasets/toxic_obfuscated_dataset.csv
"""

import pandas as pd
from pathlib import Path


def create_toxic_obfuscated_dataset():
	"""Create dataset with obfuscated toxic content"""

	# Toxic examples with various obfuscation techniques
	toxic_texts = [
		# F-word variations
		"F*ck you and your stupid ideas",
		"What the f**k is wrong with you",
		"F*** off you idiot",
		"This is f*cking bullsh*t",
		"Go f*** yourself",
		"You f**king moron",
		"F ck this garbage",
		"F.u.c.k you",
		"Fck you loser",
		"F-word you very much",
		"Eff you and die",
		"Fvck off",
		"Phuck you",
		"Fuk you",
		"F@ck you",
		"F#ck this",

		# S-word variations
		"You piece of sh*t",
		"This is sh*t quality",
		"You're full of sh**",
		"Eat sh*t and die",
		"Holy sh*t that's bad",
		"You sh*thead",
		"Sh1t for brains",
		"What a sh*tshow",
		"Sh!t post",
		"Sht up",
		"Sh.it content",
		"Shlt",
		"Shiet",
		"Shyt",
		"Sh@t",

		# B-word variations
		"You stupid b*tch",
		"What a b**ch move",
		"Son of a b*tch",
		"Stop being a b*tch",
		"B1tch please",
		"Shut up b!tch",
		"B.i.t.c.h",
		"Btch",
		"Betch",
		"Biatch",
		"B@tch",
		"B!+ch",

		# A-word variations
		"You're such an a**hole",
		"What an @sshole",
		"A** wipe",
		"Kiss my a**",
		"You dumb@ss",
		"A**hat",
		"A-hole",
		"Azzhole",
		"A$$hole",
		"@ss",
		"A.s.s",

		# D-word variations
		"You're a d*ck",
		"Stop being a d**k",
		"D*ckhead",
		"What a d!ck move",
		"D1ck",
		"D!ck",
		"Dck",
		"Dik",

		# Hate speech with obfuscation
		"I h@te you so much",
		"I h8te everything about you",
		"H8 you forever",
		"Full of h@tred",
		"I h.a.t.e. you",
		"Hte you",

		# Death threats obfuscated
		"K*ll yourself",
		"Go k!ll yourself",
		"K1ll yourself idiot",
		"Kll yourself",
		"K.i.l.l yourself",
		"End your l!fe",
		"D!e in a fire",
		"I hope you d1e",
		"Drop d3ad",
		"D.i.e. slowly",

		# Slurs with obfuscation
		"You're a r*tard",
		"Such a ret@rd",
		"R3tard",
		"Ret@rded idiot",
		"You ret4rd",

		# Multiple obfuscations
		"F**k off you stupid @ss b*tch",
		"You're a f*cking ret@rd",
		"Sh*t for brains @sshole",
		"D*ckhead piece of sh*t",
		"F**k you and d!e you b1tch",
		"H8 you so f*cking much",
		"K!ll yourself you worthless sh*t",

		# Leet speak obfuscation
		"F4ck you",
		"Y0u suck",
		"5tupid 1diot",
		"D13 in a fire",
		"H8r gonna h8",
		"B1tch @ss",

		# Unicode/special chars
		"Fųck you",
		"Shït head",
		"Bîtch",
		"Àsshole",

		# Spacing obfuscation
		"F u c k  you",
		"S h i t post",
		"B i t c h move",
		"D i e  slow",

		# Creative masking
		"F[censored]k you",
		"Sh[bleep]t content",
		"B[redacted]tch",
		"You piece of [censored]",

		# Compound toxic
		"You're a f*cking waste of sp@ce",
		"Die slow you worthless sh*t",
		"K!ll yourself you f**king @sshole",
		"I h8te you so f*cking much b*tch",
		"Go to h3ll you ret@rded piece of garbage",

		# Severe toxic (heavily obfuscated)
		"I hope you f*cking d1e painfully",
		"You should k!ll yourself you worthless sh*t",
		"Everyone h@tes you, just d!e already",
		"F**k off and never come back @sshole",
		"You're absolutely f*cking pathetic trash",
		"Die in a f*re you stupid b*tch",
		"K!ll yourself n0body likes you",
		"I hope terrible th1ngs happen to you",
		"You deserve to d1e slowly",
		"The world would be better without you, d!e",
	]

	# Non-toxic examples (for balance)
	clean_texts = [
		"I love this product, amazing quality!",
		"Great article, very informative and helpful",
		"Thank you so much for your help!",
		"Excellent work on this project",
		"This is really helpful information",
		"Wonderful explanation, very clear",
		"You are kind and thoughtful",
		"Great job, keep up the excellent work!",
		"I appreciate your assistance",
		"This is exactly what I needed",
		"Beautiful work, well done",
		"Very informative and useful content",
		"I really appreciate this, thank you",
		"Fantastic job on this",
		"Well researched and documented",
		"You're awesome, thanks!",
		"This helped me a lot",
		"Super helpful, much appreciated",
		"Amazing quality and service",
		"Best purchase I've made",
		"The weather is nice today",
		"I had pizza for dinner",
		"The meeting is at 3pm tomorrow",
		"This is an interesting topic",
		"I need to buy groceries later",
		"The project deadline is next week",
		"Let me check the schedule",
		"That sounds like a good plan",
		"I'll think about your suggestion",
		"Could you provide more details please?",
		"I disagree with that approach",
		"This could use some improvement",
		"That's not quite accurate",
		"I have a different perspective",
		"I'm not entirely sure about that",
		"This might need more work",
		"I have some concerns to discuss",
		"That's not my preferred method",
		"This explanation is somewhat unclear",
		"I'd like to see some changes here",
		"The movie was entertaining",
		"I enjoyed reading this book",
		"The concert was spectacular",
		"Traffic was heavy this morning",
		"I'm learning Python programming",
		"The restaurant had good food",
		"My favorite color is blue",
		"I like to exercise regularly",
		"Reading helps me relax",
		"The sunset looks beautiful tonight",
	]

	# Combine and create labels
	all_texts = toxic_texts + clean_texts
	labels = [1] * len(toxic_texts) + [0] * len(clean_texts)

	# Create DataFrame
	df = pd.DataFrame({
		'text': all_texts,
		'label': labels
	})

	# Shuffle
	df = df.sample(frac=1, random_state=42).reset_index(drop=True)

	return df


def save_dataset():
	"""Generate and save the dataset"""
	print("=" * 70)
	print("📝 GENERATING TOXIC OBFUSCATED DATASET")
	print("=" * 70)

	df = create_toxic_obfuscated_dataset()

	# Create datasets directory
	data_dir = Path('./datasets')
	data_dir.mkdir(exist_ok=True)

	# Save to CSV
	output_path = data_dir / 'toxic_obfuscated_dataset.csv'
	df.to_csv(output_path, index=False)

	# Statistics
	total = len(df)
	toxic = (df['label'] == 1).sum()
	clean = (df['label'] == 0).sum()
	toxic_pct = (toxic / total * 100)

	print(f"\n✅ Dataset created successfully!")
	print(f"\n📊 Statistics:")
	print(f"   Total samples: {total:,}")
	print(f"   Toxic: {toxic:,} ({toxic_pct:.1f}%)")
	print(f"   Clean: {clean:,} ({100 - toxic_pct:.1f}%)")
	print(f"\n💾 Saved to: {output_path}")

	# Show sample
	print(f"\n📋 Sample toxic examples (obfuscated):")
	toxic_samples = df[df['label'] == 1].head(5)
	for idx, row in toxic_samples.iterrows():
		print(f"   - {row['text']}")

	print(f"\n📋 Sample clean examples:")
	clean_samples = df[df['label'] == 0].head(5)
	for idx, row in clean_samples.iterrows():
		print(f"   - {row['text']}")

	print("\n" + "=" * 70)
	print("🚀 NEXT STEPS:")
	print("=" * 70)
	print("   1. Train model:")
	print("      python train.py --mode quick --dataset toxic_obfuscated_dataset")
	print("\n   2. Or load in your code:")
	print("      df = pd.read_csv('datasets/toxic_obfuscated_dataset.csv')")
	print("\n   3. Test the model:")
	print("      python quick_test.py")
	print("=" * 70)

	return df


if __name__ == '__main__':
	df = save_dataset()

	# Also display the actual CSV content
	print("\n" + "=" * 70)
	print("📄 CSV PREVIEW (first 10 rows):")
	print("=" * 70)
	print(df.head(10).to_string(index=False))
	print("\n... (showing 10 of {} rows)".format(len(df)))