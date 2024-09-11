import random
class DeckCard:
    shapes = [ "Hearts", "Diamonds", "Clubs", "Spades" ]
    numbers = [ "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K" ]
    
    def __init__(self):
        self.cards = [Card(number, shape) for number in DeckCard.numbers for shape in DeckCard.shapes]
    
    def __str__(self):
        return ', '.join(str(card) for card in self.cards)
    def __repr__(self):
        return self.__str__()
    def shuffle(self):
        random.shuffle(self.cards)
    def draw(self):
        return self.cards.pop() if len(self.cards) > 0 else None
        
class Card:
    def __init__(self, number, shape):
        self.number = number
        self.shape = shape

    def __str__(self):
        return f"{self.number} {self.shape}"
    def __repr__(self):
        return self.__str__()
    
deck = DeckCard()

'''
drawing 10 cards from the deck, without replacement
'''
drawn_cards = [deck.draw() for _ in range(10) if deck.cards] or print("No more cards to draw")
print(drawn_cards)

