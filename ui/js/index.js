let firstQuery = true;
let retrieval_active = false;

document.addEventListener('DOMContentLoaded', event => {
	const form = document.getElementById('query-form');
	const queryInput = document.getElementById('query');
	const chathistory = document.querySelector('.history');
	let query = '';
	const blur = document.querySelector('.blur');
	const retrievalButton = document.querySelector('.retrieval_button');
	const retrievalResult = document.querySelector('.retrieval_result');
	const retrievalFocused = document.querySelector('.retrieval_result_focused');

	form.addEventListener('submit', function (event) {
		event.preventDefault(); // Prevent the default form submission
		query = queryInput.value;
		queryInput.value = '';
		if (firstQuery) {
			form.classList.add('moved');
			retrievalButton.classList.add('retrieval_button--active');
			chathistory.classList.add('history--active');
			firstQuery = !firstQuery;
		}
		chathistory.innerHTML += `<p class="message user_message">${query}</p>`;
	});

	retrievalButton.addEventListener('click', () => {
		retrievalResult.classList.toggle('retrieval_result--active');
	});

	const socket = io('http://localhost:5000');

	socket.on('ir_results', data => {
		retrievalButton.classList.toggle('retrieval_button--rotate');
		retrievalResult.innerHTML = `${data.ir_results
			.map(result => `<div class="retrieved_item" tabindex="0">${result.question}</div>`)
			.join('')}`;
		retrievalFocused.innerHTML = `${data.ir_results
			.map(
				result =>
					`<table class="retrieved_item_table"><tr><td>سوال</td><td>${result.question}</td></tr><tr><td>پاسخ</td><td>${result.answer}</td></tr><tr><td>دسته‌بندی</td><td>${result.category}</td></tr></table>`
			)
			.join('')}`;

		addEventListenersToRetrievedItems();
	});

	socket.on('llm_response', data => {
		chathistory.innerHTML += `<p class="message agent_message">${data.llm_response}</p>`;
	});

	form.addEventListener('submit', async e => {
		e.preventDefault();
		console.log(query);
		const response = await fetch('/submit', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*',
			},
			body: JSON.stringify({ query }),
		});
		if (!response.ok) {
			console.error('Failed to submit query');
		}
	});

	function addEventListenersToRetrievedItems() {
		let retrievedItems = document.querySelectorAll('.retrieved_item');
		let currentIndex = -1;

		// Remove any existing event listeners to avoid duplicates
		retrievedItems.forEach(item => {
			item.removeEventListener('click', handleItemClick);
			item.removeEventListener('focus', handleItemFocus);
			item.removeEventListener('blur', handleItemBlur);
		});

		retrievedItems.forEach((item, index) => {
			item.addEventListener('click', () => handleItemClick(item, index));
			item.addEventListener('focus', () => handleItemFocus(index));
			item.addEventListener('blur', () => handleItemBlur(index));
		});

		document.addEventListener('keydown', event => {
			if (currentIndex === -1) return; // If no item has been clicked yet, do nothing

			if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
				// Move to the next or previous item
				if (event.key === 'ArrowUp') {
					currentIndex = currentIndex > 0 ? currentIndex - 1 : retrievedItems.length - 1;
				} else if (event.key === 'ArrowDown') {
					currentIndex = currentIndex < retrievedItems.length - 1 ? currentIndex + 1 : 0;
				}

				// Deselect the previously selected item
				if (document.activeElement.classList.contains('retrieved_item')) {
					document.activeElement.blur(); // Remove focus from the previous item
				}

				// Select the new item
				const selectedItem = retrievedItems[currentIndex];
				selectedItem.focus(); // This will make the div "active" so it can be styled differently if needed
				// resizeDiv(selectedItem); // Resize the newly selected item
			}

			// if (event.key == "Escape") {
			// 	currentIndex
			// }
		});
	}

	function handleItemClick(item, index) {
		// resizeDiv(item);
		// Update current index
		currentIndex = index;
		// Focus the clicked item
		item.focus();
	}

	function handleItemFocus(index) {
		document
			.querySelectorAll('.retrieved_item_table')
			[index].classList.add('retrieved_item_table--active');

		if (!blur.classList.contains('blured')) {
			retrievalFocused.classList.add('retrieval_result_focused--active');
			blur.classList.add('blured');
		}
	}

	function handleItemBlur(index) {
		// Logic to execute when the item loses focus
		const currentItemTable = document.querySelectorAll('.retrieved_item_table')[index];

		// Add class to indicate that the item is being blurred
		currentItemTable.classList.remove('retrieved_item_table--active');

		// Set a flag to track if another item has gained focus
		let focusGained = false;

		// Function to check if focus has moved to another item
		const checkFocus = () => {
			const activeElement = document.activeElement;
			const retrievedItems = document.querySelectorAll('.retrieved_item');

			// Check if the currently focused element is one of the retrieved items
			for (let item of retrievedItems) {
				if (item === activeElement) {
					focusGained = true;
					break;
				}
			}

			// If no item has gained focus, execute special logic
			if (!focusGained) {
				retrievalFocused.classList.remove('retrieval_result_focused--active');
				blur.classList.remove('blured');
			}
		};

		// Use setTimeout to allow the blur event to complete before checking focus
		setTimeout(checkFocus, 0);
	}

	function resizeDiv(div) {
		const currentHeight = div.offsetHeight;
		const newHeight = currentHeight === 100 ? 200 : 100; // Toggle between two sizes
		div.style.height = `${newHeight}px`;
	}
});
